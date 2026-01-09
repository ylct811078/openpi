"""为 LeRobot 数据集添加归一化价值标签。

价值计算公式：value = -(T - t) / T
- t: 当前帧索引
- T: episode 总帧数
- value 范围: [-1, 0]

用法:
    python scripts/add_value_labels.py --data_dir /path/to/lerobot_dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_value_labels(episode_length: int) -> np.ndarray:
    """计算一个 episode 的归一化价值标签。
    
    Args:
        episode_length: 总帧数 T
        
    Returns:
        value_labels: [T], 范围 [-1, 0]
    """
    T = episode_length
    t = np.arange(T)
    value_normalized = -(T - t) / T
    return value_normalized.astype(np.float32)


def add_value_to_parquet(parquet_path: Path) -> int:
    """为单个 parquet 文件添加 value 列。
    
    Args:
        parquet_path: parquet 文件路径
        
    Returns:
        帧数
    """
    df = pd.read_parquet(parquet_path)
    
    if 'value' in df.columns:
        print(f"跳过 {parquet_path.name}: value 列已存在")
        return len(df)
    
    episode_length = len(df)
    value_labels = compute_value_labels(episode_length)
    
    df['value'] = value_labels
    
    df.to_parquet(parquet_path, index=False)
    
    return episode_length


def update_info_json(info_path: Path):
    """更新 info.json，添加 value 字段信息。"""
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    if 'features' in info and 'value' in info['features']:
        print("info.json 已包含 value 字段")
        return
    
    if 'features' not in info:
        info['features'] = {}
    
    info['features']['value'] = {
        'dtype': 'float32',
        'shape': [1],
        'description': 'Normalized value label: -(T-t)/T, range [-1, 0]'
    }
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"已更新 {info_path}")


def main():
    parser = argparse.ArgumentParser(description='为 LeRobot 数据集添加价值标签')
    parser.add_argument('--data_dir', type=str, required=True, help='LeRobot 数据集路径')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    parquet_dir = data_dir / 'data'
    if not parquet_dir.exists():
        raise ValueError(f"找不到 data 目录: {parquet_dir}")
    
    parquet_files = sorted(parquet_dir.rglob('*.parquet'))
    
    if not parquet_files:
        raise ValueError(f"找不到 parquet 文件: {parquet_dir}")
    
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    total_frames = 0
    for parquet_path in tqdm(parquet_files, desc="处理 episodes"):
        frames = add_value_to_parquet(parquet_path)
        total_frames += frames
    
    info_path = data_dir / 'meta' / 'info.json'
    if info_path.exists():
        update_info_json(info_path)
    
    print(f"\n完成! 共处理 {len(parquet_files)} 个 episodes, {total_frames} 帧")
    print(f"每帧添加了 value 字段，范围 [-1, 0]")


if __name__ == '__main__':
    main()
