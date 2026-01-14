# 价值函数模型

## 概述

本文档描述了结合 SigLIP (400M) 和 Gemma3 (270M) 的价值函数 VLM（视觉语言模型）实现，用于强化学习中的价值估计。采用 C51 分布式强化学习的方法进行价值预测，并使用**交叉注意力机制**实现图像和文本的深度融合。

## 架构

### 整体架构图

```
                    +------------------+
                    |     输入图像      |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  SigLIP So400m   |
                    |  (4亿参数)        |
                    |  输出: 1152维     |
                    +--------+---------+
                             |
                             | img_projection
                             | (1152 → 640)
                             v
                    +------------------+
                    |   图像特征        |
                    |   [B, seq_img, 640]
                    +--------+---------+
                             |
                             | (作为 Key/Value)
                             |
+------------------+         v
|   任务文本        |   +------------------+
+--------+---------+   |   交叉注意力层     |
         |             |   8 heads, 640维  |
         v             +--------+---------+
+------------------+         |
| Gemma3 Embedder  |         | (Query来自文本)
| (冻结权重)        |         |
+--------+---------+         v
         |             +------------------+
         |             |  增强的文本特征   |
         +------------>|  (残差连接)      |
                       +--------+---------+
                                |
                                v
                       +------------------+
                       | 拼接图像+文本特征 |
                       | [B, seq_total, 640]
                       +--------+---------+
                                |
                                | 输入 Backbone (Deep Fusion)
                                v
                       +------------------+
                       | Gemma Transformer|
                       |     (18 Layers)  |
                       | [B, seq_total, 640]
                       +--------+---------+
                                |
                                | 加权平均池化
                                v
                       +------------------+
                       |   Value Head     |
                       | LayerNorm + MLP  |
                       | 640 → 320 → 201  |
                       +--------+---------+
                                |
                                v
                       +------------------+
                       |  价值分布 (201)   |
                       |  Softmax → 期望值 |
                       +------------------+
```

### 交叉注意力与深层融合详解

**1. 交叉注意力 (Cross-Attention)**:
- 文本特征作为 Query，图像特征作为 Key/Value。
- 让文本 Token 能够"关注"并聚合相关的图像区域特征。
- 此时特征仅进行了初步的模态交互。

**2. 深层融合 (Deep Fusion via Backbone)**:
- 将拼接后的 [图像 + 文本] 序列送入 **Gemma Transformer Backbone**。
- 经过 **18 层** Transformer Block 的深层处理。
- 允许图像和文本信息在深层语义空间中进行复杂的推理和交互。
- 相比仅使用 Embedder 的浅层模型，具有更强的理解和泛化能力。

**3. 冻结 Embedder (Frozen Embedder)**:
- 文本 Embedder 权重被**冻结** (Stop Gradient)。
- 防止在小数据集上微调时破坏预训练的词汇语义空间。
- 保持大模型的通用语言理解能力。

## C51 分布式价值函数

### 核心思想

预测价值的**分布**，然后通过期望得到价值估计。

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
   - `GemmaBlockRunner` 类：继承 Gemma3，支持 Continuous Embeddings 输入以运行 Backbone。
   - `ValueModel` 类：集成了 SigLIP, Gemma Embedder (Frozen), Cross-Attention, Gemma Backbone, Value Head。
   - `NUM_ATOMS, V_MIN, V_MAX`等：C51 分布式 RL 常量。
   - 核心逻辑：`embed_tokens` 中实现了 `Stop Gradient` (冻结) 和 `Backbone` 调用。

2. **`src/openpi/models/value_model_config.py`**
   - `ValueModelConfig`：配置类

### 修改的文件

1. **`src/openpi/training/weight_loaders.py`**
   - 修复参数加载正则 BUG：`missing_regex=".*(value_head|img_projection|cross_att).*"`，确保新层权重不被丢弃。

2. **`src/openpi/models/gemma.py`**
   - 添加 `gemma_270m` 配置。

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

### Value Head
| 参数 | 值 |
|------|-----|
| 输入维度 | 640 |
| 输出维度 | 201 (NUM_ATOMS) |

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
训练
  步骤 1：添加价值标签
  python scripts/add_value_labels.py --data_dir /home/wang/data/train_dataset/lerobot_datasets/piper_plug_libero

  步骤 2：训练模型
  python scripts/train_value.py \
      --data_dir /home/wang/data/train_dataset/lerobot_datasets/piper_plug_libero \
      --checkpoint_dir ./checkpoints/value_model \
      --batch_size 32 \
      --num_train_steps 10000 \
      --load_pretrained
