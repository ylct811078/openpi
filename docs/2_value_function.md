# 价值函数模型

## 概述

本文档描述了结合 SigLIP (400M) 和 Gemma (270M) 的价值函数 VLM（视觉语言模型）实现，用于强化学习或模仿学习中的价值估计。采用 C51 分布式强化学习的方法进行价值预测。

## 架构

```
                    +------------------+
                    |     输入图像      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  SigLIP So400m   |
                    |  (4亿参数)        |
                    |  width=1152      |
                    +--------+---------+
                             |
                             | 投影到 640 维
                             v
+------------------+   +------------------+
|   文本 Token     |   |   图像 Token     |
+--------+---------+   +--------+---------+
         |                      |
         +----------+-----------+
                    | 序列维度拼接
                    v
           +------------------+
           |   Gemma 270M     |
           |   width=640      |
           |   depth=18       |
           +--------+---------+
                    |
                    | 最后一个 token (640维)
                    v
           +------------------+
           |   Value Head     |
           |   (两层 MLP)      |
           +--------+---------+
                    |
                    v
           +------------------+
           |  价值分布 (201)   |
           |  Softmax → 期望值 |
           +------------------+
```

## Value Head 结构

Value Head 是挂载在 VLM 骨干网络末端的两层 MLP 结构：

```
输入: [batch, 640]  ← Gemma 最后一个 token 的隐藏状态
        ↓
   LayerNorm(640)   ← 层归一化，稳定输入分布
        ↓
 Linear(640 → 320)  ← 第一层线性投影
        ↓
      GELU          ← 非线性激活
        ↓
   Dropout(0.1)     ← 防止过拟合
        ↓
 Linear(320 → 201)  ← 第二层线性投影
        ↓
输出: [batch, 201]  ← 价值分布 logits
```

| 层 | 输入维度 | 输出维度 | 说明 |
|----|---------|---------|------|
| LayerNorm | 640 | 640 | 归一化隐藏状态 |
| Linear1 | 640 | 320 | 降维投影 |
| GELU | 320 | 320 | 非线性激活 |
| Dropout | 320 | 320 | rate=0.1 |
| Linear2 | 320 | 201 | 输出 logits |

## C51 分布式价值函数

### 核心思想

不直接预测标量价值，而是预测价值的**分布**，然后通过期望得到价值估计。

### 支架设置

| 参数 | 值 |
|------|-----|
| NUM_ATOMS | 201 |
| V_MIN | -1.0 |
| V_MAX | 0.0 |
| DELTA_Z | 0.005 |
| SUPPORTS | [-1.0, -0.995, -0.99, ..., 0.0] |

### Two-hot 编码

由于真实目标值往往不会精准落在某个支架中心，采用线性插值投影：

```
目标值 y = -0.503

找到左右支架:
  b_left = 99  (对应 z = -0.505)
  b_right = 100 (对应 z = -0.500)

计算权重:
  weight_left = 0.4
  weight_right = 0.6

生成 201 维目标分布:
  P_target[99] = 0.4
  P_target[100] = 0.6
  其余位置 = 0
```

### 损失函数

交叉熵损失：

$$Loss = -\sum_{i=0}^{200} P_{target}^{(i)} \cdot \log(\text{Softmax}(L_{pred})^{(i)})$$

### 价值计算

期望值：

$$Value = \sum_{i=0}^{200} \text{Softmax}(L_{pred})^{(i)} \cdot z_i$$

## 添加/修改的文件

### 新增文件

1. **`src/openpi/models/value_model.py`**
   - `NUM_ATOMS, V_MIN, V_MAX, DELTA_Z, SUPPORTS`：C51 常量
   - `target_to_twohot()`：目标值转 two-hot 分布
   - `dist_to_value()`：分布 logits 转期望值
   - `ValueHead` 类：两层 MLP 输出头
   - `ValueModel` 类：模型主体实现
   - `embed_tokens()`：将图像和文本编码为 token 序列
   - `compute_value()`：计算期望价值
   - `compute_loss()`：计算交叉熵损失

2. **`src/openpi/models/value_model_config.py`**
   - `ValueModelConfig`：配置类

### 修改的文件

1. **`src/openpi/models/gemma.py`**
   - 添加 `gemma_270m` 配置：
     ```python
     Config(
         width=640,
         depth=18,
         mlp_dim=2048,
         num_heads=4,
         num_kv_heads=1,
         head_dim=256,
     )
     ```

## 模型参数

### Gemma 270M
| 参数 | 值 |
|------|-----|
| Width | 640 |
| Depth | 18 |
| MLP Dim | 2048 |
| Num Heads | 4 |
| Head Dim | 256 |
| 总参数量 | ~2.7亿 |

### SigLIP So400m/14
| 参数 | 值 |
|------|-----|
| Width | 1152 |
| Depth | 27 |
| Patch Size | 14x14 |
| 总参数量 | ~4亿 |

### Value Head (两层 MLP)
| 层 | 输入维度 | 输出维度 |
|----|---------|---------|
| LayerNorm | 640 | 640 |
| Linear1 | 640 | 320 |
| GELU | - | - |
| Dropout | - | - |
| Linear2 | 320 | 201 |

## 使用方法

```python
from openpi.models.value_model import ValueModel
from openpi.models.value_model_config import ValueModelConfig
import jax

# 创建配置
config = ValueModelConfig(
    dtype="bfloat16",
    gemma_variant="gemma_270m",
    siglip_variant="So400m/14",
)

# 初始化模型
model = config.create(jax.random.key(0))

# 计算价值（返回期望值）
value = model.compute_value(rng, observation, train=False)
# value shape: [batch]，范围 [-1, 0]

# 获取完整分布 logits
logits = model(observation, train=False)
# logits shape: [batch, 201]

# 计算损失（target_values 需在 [-1, 0] 范围内）
loss = model.compute_loss(rng, observation, target_values, train=True)
```

## 与 Pi0 的区别

| 方面 | Pi0 | ValueModel |
|------|-----|------------|
| 视觉编码器 | SigLIP So400m | SigLIP So400m |
| 语言模型 | Gemma 2B | Gemma 270M |
| Action Expert | Gemma 300M | 无 |
| 输出 | 动作序列 | 价值分布 (201维) |
| 损失函数 | Flow Matching | 交叉熵 (C51) |
| 总参数量 | ~30亿 | ~6.7亿 |

## 训练流程

### 1. 准备数据标签

为 LeRobot 数据集添加归一化价值标签：

```bash
python scripts/add_value_labels.py --data_dir /path/to/lerobot_dataset
```

价值计算公式：$Value = -\frac{T-t}{T}$，范围 $[-1, 0]$

### 2. 训练模型

```bash
python scripts/train_value.py \
    --data_dir /path/to/lerobot_dataset \
    --checkpoint_dir /path/to/save/checkpoints \
    --batch_size 32 \
    --num_train_steps 10000 \
    --learning_rate 1e-4 \
    --load_pretrained  # 加载 PaliGemma 预训练权重
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必填 | LeRobot 数据集路径 |
| `--checkpoint_dir` | 必填 | Checkpoint 保存路径 |
| `--batch_size` | 32 | Batch size |
| `--num_train_steps` | 10000 | 训练步数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--log_interval` | 100 | 日志间隔 |
| `--save_interval` | 1000 | 保存间隔 |
| `--num_workers` | 4 | DataLoader workers |
| `--load_pretrained` | False | 加载 PaliGemma 预训练权重 |

## 新增文件总结

| 文件 | 说明 |
|------|------|
| `src/openpi/models/value_model.py` | 模型定义 |
| `src/openpi/models/value_model_config.py` | 配置类 |
| `src/openpi/training/value_data_loader.py` | 数据加载器 |
| `src/openpi/training/weight_loaders.py` | 新增 `ValueModelWeightLoader` |
| `scripts/add_value_labels.py` | 数据标签脚本 |
| `scripts/train_value.py` | 训练脚本 |
