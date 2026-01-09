"""Value function 数据加载器。"""

from collections.abc import Iterator
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from openpi.models import model as _model
from openpi.models.value_model import target_to_twohot, NUM_ATOMS

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

        parquet_dir = self.data_dir / "data"
        self.parquet_files = sorted(parquet_dir.rglob("*.parquet"))

        if not self.parquet_files:
            raise ValueError(f"找不到 parquet 文件: {parquet_dir}")

        self.episode_lengths = []
        self.episode_offsets = [0]
        
        for pf in self.parquet_files:
            df = pd.read_parquet(pf)
            self.episode_lengths.append(len(df))
            self.episode_offsets.append(self.episode_offsets[-1] + len(df))

        self.total_frames = self.episode_offsets[-1]
        logger.info(f"加载 {len(self.parquet_files)} 个 episodes, 共 {self.total_frames} 帧")

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
        
        df = pd.read_parquet(self.parquet_files[episode_idx])
        row = df.iloc[frame_idx]

        image = self._load_image(row["image"])
        
        wrist_image = None
        if "wrist_image" in row:
            wrist_image = self._load_image(row["wrist_image"])

        state = np.array(row["state"], dtype=np.float32)

        if "value" not in row:
            raise ValueError(f"数据中没有 value 字段，请先运行 add_value_labels.py")
        value = np.float32(row["value"])

        result = {
            "image": {
                "base_0_rgb": image,
            },
            "image_mask": {
                "base_0_rgb": True,
            },
            "state": state,
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
        image_array = image_array / 255.0 * 2.0 - 1.0

        return image_array


def collate_fn(batch: list[dict]) -> dict:
    """将 batch 合并为 numpy 数组。"""
    result = {}
    
    result["image"] = {
        key: np.stack([item["image"][key] for item in batch], axis=0)
        for key in batch[0]["image"]
    }
    result["image_mask"] = {
        key: np.array([item["image_mask"][key] for item in batch], dtype=bool)
        for key in batch[0]["image_mask"]
    }
    result["state"] = np.stack([item["state"] for item in batch], axis=0)
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
        sharding: jax.sharding.Sharding | None = None,
    ):
        self.dataset = ValueDataset(data_dir, max_token_len=max_token_len)
        
        self._torch_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding

    def __len__(self) -> int:
        return len(self._torch_loader)

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray]]:
        for batch in self._torch_loader:
            batch_jax = jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                batch
            )

            observation = _model.Observation.from_dict(batch_jax)
            value = batch_jax["value"]

            yield observation, value
