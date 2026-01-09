import multiprocessing
from collections.abc import Iterator
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch

from openpi.models import model as _model

logger = logging.getLogger("openpi")


class ValueDataset(TorchDataset):
    """LeRobot 价值函数数据集。"""

    def __init__(
        self,
        data_dir: str,
        max_token_len: int = 48,
        image_size: tuple[int, int] = (224, 224),
    ):
        """
        Args:
            data_dir: LeRobot 数据集路径
            max_token_len: 最大 token 长度
            image_size: 图像大小
        """
        self.data_dir = Path(data_dir)
        self.max_token_len = max_token_len
        self.image_size = image_size

        # 加载任务文本信息
        self._load_task_texts()

        parquet_dir = self.data_dir / "data"
        self.parquet_files = sorted(parquet_dir.rglob("*.parquet"))

        if not self.parquet_files:
            raise ValueError(f"找不到 parquet 文件: {parquet_dir}")

        # 预加载所有parquet数据到内存，避免重复磁盘I/O
        logger.info("预加载所有parquet数据到内存...")
        self.episode_data = []
        self.episode_lengths = []
        self.episode_offsets = [0]

        for pf in self.parquet_files:
            df = pd.read_parquet(pf)
            self.episode_data.append(df)  # 缓存到内存
            self.episode_lengths.append(len(df))
            self.episode_offsets.append(self.episode_offsets[-1] + len(df))

        self.total_frames = self.episode_offsets[-1]
        logger.info(f"加载 {len(self.parquet_files)} 个 episodes, 共 {self.total_frames} 帧")

    def _load_task_texts(self):
        """加载任务文本信息。"""
        import json
        
        tasks_file = self.data_dir / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            raise ValueError(f"找不到任务文件: {tasks_file}")
        
        self.task_texts = {}
        with open(tasks_file, 'r') as f:
            for line in f:
                task_data = json.loads(line.strip())
                task_index = task_data["task_index"]
                task_text = task_data["task"]
                self.task_texts[task_index] = task_text
        
        logger.info(f"加载了 {len(self.task_texts)} 个任务文本")

    def _tokenize_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """简单的文本tokenization。"""
        # 添加Value:后缀，明确这是价值估计任务
        text_with_suffix = f"{text}\nValue:"
        words = text_with_suffix.lower().split()
        
        # 简单映射到数字ID（实际应该使用真正的tokenizer）
        vocab = {
            'plug': 1, 'black': 2, 'into': 3, 'the': 4, 'three': 5, 'hole': 6, 'socket': 7,
            'pull': 8, 'drawer': 9, 'open': 10, 'close': 11, 'push': 12, 'button': 13,
            'value:': 14, '\nvalue:': 14,  # Value提示符
            '<pad>': 0, '<unk>': 999
        }
        
        tokens = [vocab.get(word, vocab['<unk>']) for word in words]
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_token_len:
            tokens = tokens[:self.max_token_len]
        else:
            tokens.extend([vocab['<pad>']] * (self.max_token_len - len(tokens)))
        
        tokens = np.array(tokens, dtype=np.int32)
        mask = (tokens != vocab['<pad>']).astype(bool)
        
        return tokens, mask

    def __len__(self) -> int:
        return self.total_frames

    def _find_episode(self, idx: int) -> tuple[int, int]:
        """找到 idx 对应的 episode 和帧索引。"""
        for i, offset in enumerate(self.episode_offsets[1:], 1):
            if idx < offset:
                episode_idx = i - 1
                frame_idx = idx - self.episode_offsets[episode_idx]
                return episode_idx, frame_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> dict:
        episode_idx, frame_idx = self._find_episode(idx)

        # 使用预加载的数据，不再重复读取parquet
        row = self.episode_data[episode_idx].iloc[frame_idx]

        image = self._load_image(row["image"])

        wrist_image = None
        if "wrist_image" in row:
            wrist_image = self._load_image(row["wrist_image"])

        # 获取任务文本
        task_index = int(row["task_index"])
        task_text = self.task_texts.get(task_index, "unknown task")
        tokenized_prompt, prompt_mask = self._tokenize_text(task_text)

        # Value model 不需要机器人状态，只需要图像
        # state = np.array(row["state"], dtype=np.float32)

        if "value" not in row:
            raise ValueError("数据中没有 value 字段，请先运行 add_value_labels.py")
        value = np.float32(row["value"])

        result = {
            "image": {
                "base_0_rgb": image,
            },
            "image_mask": {
                "base_0_rgb": True,
            },
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": prompt_mask,
            # Value model 不需要 state
            # "state": state,
            "value": value,
        }

        if wrist_image is not None:
            result["image"]["wrist_0_rgb"] = wrist_image
            result["image_mask"]["wrist_0_rgb"] = True

        return result

    def _load_image(self, image_data) -> np.ndarray:
        """加载图像数据。"""
        import io

        from PIL import Image

        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError(f"未知的图像格式: {type(image_data)}")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)

        image_array = np.array(image, dtype=np.float32)
        return image_array / 255.0 * 2.0 - 1.0



def collate_fn(batch: list[dict]) -> dict:
    """将 batch 合并为 numpy 数组。"""
    result = {}

    result["image"] = {key: np.stack([item["image"][key] for item in batch], axis=0) for key in batch[0]["image"]}
    result["image_mask"] = {
        key: np.array([item["image_mask"][key] for item in batch], dtype=bool) for key in batch[0]["image_mask"]
    }
    
    # 添加文本数据处理
    result["tokenized_prompt"] = np.stack([item["tokenized_prompt"] for item in batch], axis=0)
    result["tokenized_prompt_mask"] = np.array([item["tokenized_prompt_mask"] for item in batch], dtype=bool)
    
    # Observation类需要state字段，但Value Model不使用，提供空占位符
    batch_size = len(batch)
    result["state"] = np.zeros((batch_size, 1), dtype=np.float32)  # 最小占位符
    
    result["value"] = np.array([item["value"] for item in batch], dtype=np.float32)

    return result


class ValueDataLoader:
    """价值函数数据加载器。"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        num_workers: int = 4,
        max_token_len: int = 48,
        seed: int = 42,
        sharding: jax.sharding.Sharding | None = None,
    ):
        self.dataset = ValueDataset(data_dir, max_token_len=max_token_len)
        self.seed = seed

        # π0风格的多进程优化
        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")  # 更稳定的上下文

        # π0风格的随机数生成器
        generator = torch.Generator()
        generator.manual_seed(seed)

        self._torch_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,       # π0优化
            persistent_workers=num_workers > 0,       # π0优化：保持worker存活
            collate_fn=collate_fn,
            worker_init_fn=self._worker_init_fn,      # π0优化：worker初始化
            drop_last=True,
            pin_memory=False,                         # 避免CUDA警告
            generator=generator,                      # π0优化：固定种子
        )

        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding

    def _worker_init_fn(self, worker_id: int):
        """Worker进程初始化函数，π0风格。"""
        # 设置每个worker的独立随机种子
        np.random.seed(self.seed + worker_id)
        import random
        random.seed(self.seed + worker_id)

    def __len__(self) -> int:
        return len(self._torch_loader)

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray]]:
        for batch in self._torch_loader:
            batch_jax = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

            observation = _model.Observation.from_dict(batch_jax)
            value = batch_jax["value"]

            yield observation, value
