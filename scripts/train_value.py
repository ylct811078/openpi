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


def create_train_state(
    config: ValueModelConfig,
    learning_rate: float,
    rng: at.KeyArrayLike,
    load_pretrained: bool = False,
) -> tuple[TrainState, optax.GradientTransformation]:
    """创建训练状态。"""
    model = config.create(rng)
    params = nnx.state(model)

    if load_pretrained:
        logging.info("加载 PaliGemma 预训练权重...")
        loader = ValueModelWeightLoader()
        params_dict = params.to_pure_dict()
        loaded_params = loader.load(params_dict)
        params.replace_by_pure_dict(loaded_params)
        logging.info("预训练权重加载完成")

    model_def = nnx.graphdef(model)

    tx = optax.adamw(learning_rate, weight_decay=0.01)
    opt_state = tx.init(params)

    return TrainState(
        step=0,
        params=params,
        model_def=model_def,
        opt_state=opt_state,
    ), tx


def train_step(
    state: TrainState,
    tx: optax.GradientTransformation,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    target: jnp.ndarray,
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

    new_state = TrainState(
        step=state.step + 1,
        params=new_params,
        model_def=state.model_def,
        opt_state=new_opt_state,
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }

    return new_state, info


def save_checkpoint(state: TrainState, checkpoint_dir: Path, step: int):
    """保存 checkpoint。"""
    import orbax.checkpoint as ocp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"step_{step:08d}"

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(
            ckpt_path,
            {"params": state.params.to_pure_dict(), "step": step},
        )

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
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gemma_variant", type=str, default="gemma_270m", help="Gemma 变体")
    parser.add_argument("--siglip_variant", type=str, default="So400m/14", help="SigLIP 变体")
    parser.add_argument("--load_pretrained", action="store_true", help="加载 PaliGemma 预训练权重")
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

    mesh = sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    config = ValueModelConfig(
        gemma_variant=args.gemma_variant,
        siglip_variant=args.siglip_variant,
    )

    logging.info("初始化数据加载器...")
    data_loader = ValueDataLoader(
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        sharding=data_sharding,
    )
    logging.info(f"数据集大小: {len(data_loader.dataset)} 帧")

    logging.info("初始化模型...")
    train_state, tx = create_train_state(config, args.learning_rate, init_rng, args.load_pretrained)
    logging.info("模型初始化完成")

    jax.tree.map(lambda _: replicated_sharding, train_state)

    @functools.partial(
        jax.jit,
        in_shardings=(
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
        ),
        donate_argnums=(0,),
    )
    def jit_train_step(params, model_def, opt_state, step, rng, observation, target):
        state = TrainState(step=step, params=params, model_def=model_def, opt_state=opt_state)
        new_state, info = train_step(state, tx, rng, observation, target)
        return new_state.params, new_state.model_def, new_state.opt_state, new_state.step, info

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm.tqdm(range(args.num_train_steps), dynamic_ncols=True)
    infos = []

    data_iter = iter(data_loader)

    for step in pbar:
        try:
            observation, value = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            observation, value = next(data_iter)

        with sharding.set_mesh(mesh):
            new_params, new_model_def, new_opt_state, new_step, info = jit_train_step(
                train_state.params,
                train_state.model_def,
                train_state.opt_state,
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
            )

        infos.append(info)

        if step % args.log_interval == 0 and step > 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            infos = []

        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(train_state, checkpoint_dir, step)

    save_checkpoint(train_state, checkpoint_dir, args.num_train_steps)
    logging.info("训练完成!")


if __name__ == "__main__":
    main()
