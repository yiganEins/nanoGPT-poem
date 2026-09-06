# nanoGPT 中文古诗生成

一个基于 PyTorch 从零实现的 GPT（Decoder-only Transformer）模型，用于**中文古诗生成**。参考 [Karpathy 的 nanoGPT](https://github.com/karpathy/nanoGPT) 思路，使用 SentencePiece 分词，不依赖 HuggingFace Transformers。

## 项目结构

```
├── main.py              # 训练 + 文本生成入口
├── nanoGPT/
│   └── model.py         # GPT 模型定义（Attention / Block / GPT）
├── data/
│   ├── get_data.py      # 数据加载、批处理、SentencePiece 分词器封装
│   └── tokenize.py      # 训练 SentencePiece 分词器
└── log/                 # 训练日志与模型权重（训练时自动生成，不随仓库上传）
```

## 模型结构

- Decoder-only Transformer，采用前置归一化（Pre-Norm）架构
- 嵌入层与输出层**权重共享**（weight tying）
- 默认超参数（在 `main.py` 顶部可改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 256 | 词向量维度 |
| `nums_layer` | 10 | Transformer 层数 |
| `nums_head` | 8 | 多头注意力头数 |
| `dropout` | 0.2 | Dropout 比例 |
| `max_len` | 256 | 序列最大长度 |
| `batch_size` | 64 | 批大小 |
| `iters` | 8000 | 训练迭代数 |
| `learning_rate` | 0.0006 | 初始学习率 |

## 依赖

- Python 3.x
- [PyTorch](https://pytorch.org/)（建议 1.13+）
- [sentencepiece](https://github.com/google/sentencepiece)
- numpy
- pandas

```bash
pip install torch sentencepiece numpy pandas
```

## 快速开始

### 1. 准备数据

训练数据是一个纯文本文件 `data/poems.ch`，**每行一首古诗**，例如：

```
床前明月光，疑是地上霜。
花有重开日，人无再少年。
...
```

> 注意：`poems.ch` 体积较大，未包含在仓库中，如果需要训练模型请自行准备语料。

### 2. 训练分词器

用 SentencePiece 在语料上训练一个 BPE 分词器（词表大小 12000），生成 `data/input_poems.model`，`data/input_poems.vocab`

### 3. 训练模型

在 `main.py` 底部把 `train()` 取消注释：

```python
if __name__ == '__main__':
    train() # 训练模型
    # test() # 推理生成
```

然后运行：

```bash
python main.py
```

训练过程会：
- 自动按 9:1 划分训练集 / 验证集
- 使用 AdamW 优化器 + 余弦退火学习率（含 warmup）
- 每 200 步评估一次 loss，**只保存验证集上最优的模型**
- 早停（验证 loss 连续 10 次无提升即停止）
- 将日志与权重保存到 `log/dm-256_nl-10_nh-8_lr-0.0006_it-8000/`：
  - `GPT.pt` —— 最优模型权重
  - `loss.tsv` —— 训练 / 验证 loss 与学习率记录（可用 Excel / matplotlib 绘制）
  - `config.json` —— 本次训练的超参数

### 4. 生成文本

训练完成后，在 `main.py` 底部改回 `test()`，运行：

```bash
python main.py
```


## 核心代码说明

- `nanoGPT/model.py`
  - `Attention`：单头自注意力（因果掩码）
  - `MultiHeadAttention`：多头自注意力
  - `FeedForward`：两层前馈网络（GELU）
  - `Block`：一个 Decoder 块（前置归一化 + 残差）
  - `GPT`：整体模型，含 `forward` 与 `generate`（支持 temperature、top-k 采样、`eos` 停止）
- `data/get_data.py`
  - `Data`：加载语料、编码、随机批采样（`get_batch`）
  - `Tokenizer`：封装 SentencePiece，提供 `encode` / `decode`
- `data/tokenize.py`
  - 训练 SentencePiece 分词器，以及分词结果的小测试