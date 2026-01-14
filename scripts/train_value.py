"""Value function 训练脚本。

用法:
    python scripts/train_value.py \
        --data_dir /path/to/lerobot_dataset \
        --checkpoint_dir /path/to/save/checkpoints \
        --batch_size 32 \
        --num_train_steps 10000 \
        --load_pretrained  # 加载 PaliGemma 预训练权重
"""

import argparse
import dataclasses
import functools
import logging
from pathlib import Path
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm

from openpi.models import model as _model
from openpi.models.value_model_config import ValueModelConfig
from openpi.shared import array_typing as at
import openpi.training.sharding as sharding
from openpi.training.value_data_loader import ValueDataLoader
from openpi.training.weight_loaders import ValueModelWeightLoader


def init_logging():
    """Custom logging format."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


@dataclasses.dataclass
class TrainState:
    step: int
    params: nnx.State
    model_def: nnx.GraphDef
    opt_state: optax.OptState
    ema_params: nnx.State | None = None  


def create_train_state(
    config: ValueModelConfig,
    num_train_steps: int,
    rng: at.KeyArrayLike,
    load_pretrained: bool = False,
    ema_decay: float = 0.999,  
) -> tuple[TrainState, optax.GradientTransformation]:
    """创建训练状态。"""
    model = config.create(rng)
    params = nnx.state(model)

    if load_pretrained:
        logging.info("加载 SigLIP + Gemma3-270M 预训练权重...")
        loader = ValueModelWeightLoader()
        params_dict = params.to_pure_dict()
        loaded_params = loader.load(params_dict)
        params.replace_by_pure_dict(loaded_params)
        logging.info("预训练权重加载完成")

    model_def = nnx.graphdef(model)

    
    warmup_steps = min(1000, num_train_steps // 10)  # 前10%步数用于warmup
    
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.0,
                end_value=5e-5,  
                transition_steps=warmup_steps,
            ),
            optax.cosine_decay_schedule(
                init_value=5e-5,
                decay_steps=num_train_steps - warmup_steps,
                alpha=0.1, 
            )
        ],
        [warmup_steps]
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # 梯度裁剪
        optax.adamw(
            lr_schedule, 
            b1=0.9,        # π0优化：更保守的momentum
            b2=0.95,       # π0优化：更保守的二阶momentum
            eps=1e-8,      # π0优化：数值稳定性
            weight_decay=1e-10  # π0优化：极小权重衰减，避免OOM
        )
    )
    opt_state = tx.init(params)

    return TrainState(
        step=0,
        params=params,
        model_def=model_def,
        opt_state=opt_state,
        ema_params=jax.tree.map(lambda x: x, params) if ema_decay is not None else None,  # π0优化：初始化EMA
    ), tx


def train_step(
    state: TrainState,
    tx: optax.GradientTransformation,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    target: jnp.ndarray,
    ema_decay: float = 0.999,  # π0优化：EMA衰减率
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """单步训练。"""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model):
        return model.compute_loss(rng, observation, target, train=True)

    jax.random.fold_in(rng, state.step)
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # π0优化：EMA更新
    new_ema_params = None
    if state.ema_params is not None:
        new_ema_params = jax.tree.map(
            lambda old, new: ema_decay * old + (1 - ema_decay) * new, 
            state.ema_params, 
            new_params
        )

    new_state = TrainState(
        step=state.step + 1,
        params=new_params,
        model_def=state.model_def,
        opt_state=new_opt_state,
        ema_params=new_ema_params,
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }

    return new_state, info


def load_checkpoint(checkpoint_path: Path, state: TrainState) -> TrainState:
    """从 checkpoint 恢复训练状态。"""
    import orbax.checkpoint as ocp
    
    # 确保使用绝对路径
    checkpoint_path = checkpoint_path.resolve()
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint 不存在: {checkpoint_path}")
    
    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(str(checkpoint_path))
    
    # 恢复参数
    state.params.replace_by_pure_dict(restored["params"])
    
    # 恢复 EMA 参数（如果存在）
    if "ema_params" in restored and state.ema_params is not None:
        state.ema_params.replace_by_pure_dict(restored["ema_params"])
    
    # 恢复步数
    restored_step = int(restored["step"])
    state = TrainState(
        step=restored_step,
        params=state.params,
        model_def=state.model_def,
        opt_state=state.opt_state,
        ema_params=state.ema_params,
    )
    
    logging.info(f"从 checkpoint 恢复: {checkpoint_path}, step={restored_step}")
    return state


def save_checkpoint(state: TrainState, checkpoint_dir: Path, step: int):
    """保存 checkpoint。"""
    import orbax.checkpoint as ocp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 确保使用绝对路径
    ckpt_path = (checkpoint_dir / f"step_{step:08d}").resolve()

    with ocp.PyTreeCheckpointer() as ckptr:
        save_dict = {
            "params": state.params.to_pure_dict(), 
            "step": step
        }
        # π0优化：同时保存EMA参数
        if state.ema_params is not None:
            save_dict["ema_params"] = state.ema_params.to_pure_dict()
        
        ckptr.save(str(ckpt_path), save_dict)

    logging.info(f"保存 checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="训练 Value Function")
    parser.add_argument("--data_dir", type=str, required=True, help="LeRobot 数据集路径")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint 保存路径")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_train_steps", type=int, default=10000, help="训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存间隔")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gemma_variant", type=str, default="gemma_270m", help="Gemma 变体")
    parser.add_argument("--siglip_variant", type=str, default="So400m/14", help="SigLIP 变体")
    parser.add_argument("--fsdp_devices", type=int, default=1, help="FSDP设备数量，>1启用模型并行")
    parser.add_argument("--load_pretrained", action="store_true", help="加载 PaliGemma 预训练权重")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定checkpoint恢复训练（例如：step_00001000）")
    args = parser.parse_args()

    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 配置JAX
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")  # 优先使用CUDA，fallback到CPU

    # 设置JAX配置
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    try:
        logging.info(f"JAX devices: {jax.devices()}")
        logging.info(f"JAX default backend: {jax.default_backend()}")
    except Exception as e:
        logging.warning(f"JAX设备检测失败: {e}")
        logging.info("继续使用CPU进行训练")

    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)

    # π0风格多GPU配置 - 自动使用所有可用GPU
    available_devices = jax.device_count()
    # 如果用户没有指定fsdp_devices，自动使用所有GPU
    if args.fsdp_devices == 1 and available_devices > 1:
        args.fsdp_devices = available_devices
        logging.info(f"自动调整：使用所有 {available_devices} 个GPU进行FSDP")
    
    logging.info(f"可用设备数: {available_devices}, FSDP设备数: {args.fsdp_devices}")
    if available_devices % args.fsdp_devices != 0:
        raise ValueError(f"设备数 {available_devices} 必须能被FSDP设备数 {args.fsdp_devices} 整除")
    
    mesh = sharding.make_mesh(num_fsdp_devices=args.fsdp_devices)
    logging.info(f"Mesh形状: {mesh.shape}, 轴: {mesh.axis_names}")
    
    # 优化后的sharding配置
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    config = ValueModelConfig(
        gemma_variant=args.gemma_variant,
        siglip_variant=args.siglip_variant,
    )

    logging.info("初始化数据加载器...")
    # 增加batch size以提高GPU利用率
    effective_batch_size = args.batch_size * max(1, available_devices // args.fsdp_devices)
    logging.info(f"原始batch size: {args.batch_size}, 有效batch size: {effective_batch_size}")
    
    # 调整worker数量，避免过多worker导致的问题
    max_workers = min(6, max(1, args.num_workers))  # 系统建议最大6个worker
    
    data_loader = ValueDataLoader(
        args.data_dir,
        batch_size=effective_batch_size,  # 使用更大的有效batch size
        shuffle=True,
        num_workers=max_workers,  # 使用调整后的worker数量
        sharding=None,  # 先不传递sharding，避免序列化问题
    )
    # 初始化后设置sharding，避免多进程序列化问题
    data_loader.set_sharding(data_sharding)
    logging.info(f"数据集大小: {len(data_loader.dataset)} 帧")

    logging.info("初始化模型...")
    train_state, tx = create_train_state(config, args.num_train_steps, init_rng, args.load_pretrained)
    logging.info("模型初始化完成")
    
    # 从检查点恢复（如果指定）
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.checkpoint_dir) / args.resume_from_checkpoint
        train_state = load_checkpoint(checkpoint_path, train_state)
        logging.info(f"从 step {train_state.step} 继续训练")

    # 将模型参数分片到多GPU
    if args.fsdp_devices > 1:
        logging.info("应用FSDP分片到模型参数...")
        with sharding.set_mesh(mesh):
            train_state = jax.tree_map(
                lambda x: sharding.apply_fsdp_sharding(mesh, x) if hasattr(x, 'shape') else x,
                train_state,
                is_leaf=lambda x: hasattr(x, 'shape')
            )
        logging.info("FSDP分片完成")
    else:
        # 单卡：复制到所有设备
        train_state = jax.tree_map(
            lambda x: jax.device_put(x, replicated_sharding), 
            train_state,
            is_leaf=lambda x: hasattr(x, 'shape')
        )

    @functools.partial(
        jax.jit,
        in_shardings=(
            replicated_sharding,  # params
            replicated_sharding,  # model_def
            replicated_sharding,  # opt_state
            replicated_sharding,  # ema_params
            replicated_sharding,  # step
            replicated_sharding,  # rng
            data_sharding,        # observation
            data_sharding,        # target
        ),
        out_shardings=(
            replicated_sharding,  # new_params
            replicated_sharding,  # new_model_def
            replicated_sharding,  # new_opt_state
            replicated_sharding,  # new_ema_params
            replicated_sharding,  # new_step
            replicated_sharding,  # info
        ),
        # 移除donate_argnums，避免缓冲区重复使用错误
    )
    def jit_train_step(params, model_def, opt_state, ema_params, step, rng, observation, target):
        state = TrainState(
            step=step, 
            params=params, 
            model_def=model_def, 
            opt_state=opt_state,
            ema_params=ema_params
        )
        new_state, info = train_step(state, tx, rng, observation, target)
        return new_state.params, new_state.model_def, new_state.opt_state, new_state.ema_params, new_state.step, info

    checkpoint_dir = Path(args.checkpoint_dir).resolve()  # 确保绝对路径
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 从恢复的步数开始训练
    start_step = train_state.step
    total_steps = args.num_train_steps
    pbar = tqdm.tqdm(range(start_step, total_steps), initial=start_step, total=total_steps, dynamic_ncols=True)
    infos = []

    # π0风格：数据预取和GPU利用率优化
    data_iter = iter(data_loader)
    
    # 预取多个batch，减少GPU等待
    logging.info("预取前几个batch以优化GPU利用率...")
    prefetch_batches = []
    for i in range(min(3, len(data_loader))):  # 预取3个batch
        try:
            batch = next(data_iter)
            prefetch_batches.append(batch)
        except StopIteration:
            break
    
    if not prefetch_batches:
        logging.error("无法预取任何batch，检查数据集")
        return
        
    logging.info(f"成功预取 {len(prefetch_batches)} 个batch")

    # JIT预热：避免首次编译的GPU空闲
    logging.info("JIT编译预热...")
    observation, value = prefetch_batches[0]
    with sharding.set_mesh(mesh):
        _ = jit_train_step(
            train_state.params,
            train_state.model_def,
            train_state.opt_state,
            train_state.ema_params,
            train_state.step,
            train_rng,
            observation,
            value,
        )
    logging.info("JIT编译完成，开始训练...")
    
    # 重新初始化数据迭代器
    data_iter = iter(data_loader)
    prefetch_idx = 0

    for step in pbar:
        # 使用预取的batch或获取新batch
        if prefetch_idx < len(prefetch_batches):
            observation, value = prefetch_batches[prefetch_idx]
            prefetch_idx += 1
        else:
            try:
                observation, value = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                observation, value = next(data_iter)
        with sharding.set_mesh(mesh):
            new_params, new_model_def, new_opt_state, new_ema_params, new_step, info = jit_train_step(
                train_state.params,
                train_state.model_def,
                train_state.opt_state,
                train_state.ema_params,
                train_state.step,
                train_rng,
                observation,
                value,
            )
            train_state = TrainState(
                step=new_step,
                params=new_params,
                model_def=new_model_def,
                opt_state=new_opt_state,
                ema_params=new_ema_params,
            )

        infos.append(info)

        if step % args.log_interval == 0 and step > 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            
            # 获取当前学习率
            current_lr = tx.learning_rate if hasattr(tx, 'learning_rate') else 'N/A'
            if hasattr(tx, 'inner') and hasattr(tx.inner, 'learning_rate'):
                current_lr = tx.inner.learning_rate
            
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}, lr={current_lr:.2e}" if isinstance(current_lr, (int, float)) else f"Step {step}: {info_str}")
            infos = []

        if step % args.save_interval == 0 and step > 0:
            # 计算当前损失
            current_loss = jax.device_get(info["loss"]).item()
            logging.info(f"保存检查点 - Step {step}, Loss: {current_loss:.4f}")
            save_checkpoint(train_state, checkpoint_dir, step)

    save_checkpoint(train_state, checkpoint_dir, args.num_train_steps)
    logging.info("训练完成!")


if __name__ == "__main__":
    main()
