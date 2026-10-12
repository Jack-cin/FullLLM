# FullLLM

一个从零复现「小语言模型完整流程」的项目：**TinyStories 数据 → 训练分词器 → 预训练 Transformer → PPO 微调 → 导出/部署 → 采样**。
模型结构是 LLaMA 风格（RMSNorm + RoPE + SwiGLU + GQA + KV Cache），训练脚本是 nanoGPT / llama2.c 的改法，全部代码带中文逐行注释。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `tinystories.py` | 数据流水线：下载 → 训练分词器 → 预分词（pretokenize）→ `DataLoader` |
| `tokenizer.py` | SentencePiece 分词器封装（encode/decode/pad/导出 llama2.c 格式 `.bin`） |
| `model.py` | Transformer（RoPE + GQA + KV Cache）、`TransformerWithValueHead`（PPO 用）、采样函数 |
| `train.py` | 预训练 / 续训主循环（AMP、梯度累积、余弦学习率、DDP、`torch.compile`） |
| `ppo.py` | PPO 训练器（`PPOTrainer.step()`：rollout 评估 → GAE → clip 损失 → 更新） |
| `export.py` | 把 checkpoint 导出成 llama2.c 格式（v0/v1/v2 量化）/ HF 格式 |
| `sample.py` | 加载 checkpoint 做文本生成 |
| `tests/smoke_test.py` | 冒烟测试：无 GPU、无需原始数据集，用合成数据跑通全链路 |

## 0. 环境准备

```bash
pip install -r requirements.txt        # torch / sentencepiece / numpy / tqdm / tyro / pytest
```

> `--device=cpu` 也能跑（本仓库的测试就是在纯 CPU 上做的）；`--dtype=float32` 仅在 CPU 下有意义，
> GPU 上用 `float16`（配 GradScaler）或 `bfloat16`。

## 1. 数据准备

```bash
# ① 下载 TinyStories 原始数据（约 1.5GB，落到 data/TinyStories_all_data/）
python tinystories.py download

# ② 训练 512 词表的自定义分词器（输出 data/tok512.model）
python tinystories.py train_vocab --vocab_size=512

# ③ 预分词：把 json 分片转成 uint16 的 .bin（输出 data/tok512/*.bin）
python tinystories.py pretokenize --vocab_size=512
```

想用 Llama2 官方 32k 词表：跳过分词器训练，直接 `python tinystories.py pretokenize`（不加 `--vocab_size`），
并把训练参数换成 `--vocab_source=llama2 --vocab_size=32000`。

## 2. 预训练

```bash
python train.py \
    --out_dir="outmini" \
    --batch_size=128 --max_seq_len=512 --gradient_accumulation_steps=16 \
    --vocab_source=custom --vocab_size=512 \
    --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 --multiple_of=4 \
    --learning_rate=1e-3 --dropout=0.05 --weight_decay=0.01 \
    --max_iters=100000 --beta2=0.99 --warmup_iters=1000 \
    --eval_interval=2000 --eval_iters=100 --compile \
    --device=cuda --dtype=bfloat16 --num_workers=4
```

产物：`outmini/ckpt.pt`（模型 + 优化器 + 配置）、`outmini/model.bin`（llama2.c 格式，可给 C 推理程序用）。

## 3. 续训 / PPO 微调

```bash
# 续训（自动在 out_dir 里找 ckpt.pt / ppo_ckpt.pt，也可用 --ckpt_path 显式指定）
python train.py --init_from=resume --out_dir="outmini" --device=cuda --ppo \
    --learning_rate=1e-4 --max_iters=200000 --always_save_checkpoint
```

> ⚠️ **注意当前状态**：`--ppo` 只是把模型换成 `TransformerWithValueHead`（多一个 value head）、
> 并把 checkpoint 存成 `ppo_ckpt.pt`，训练目标仍然是**交叉熵**——仓库里还没有把 `ppo.py` 的
> rollout/奖励/KL 串进 `train.py`。`PPOTrainer.step()` 本身是可用的（见 `tests/smoke_test.py::test_ppo_step_runs`），
> 需要自己写 rollout：采样 prompt → `model.generate_withcache()` 生成 response → 用 `LengthReward`
> 之类打分 → 调用 `PPOTrainer.step(queries, responses, scores)`。

## 4. 采样与导出

```bash
# 生成（--top_p / --top_k / --temperature 可调）
python sample.py --checkpoint=outmini/ckpt.pt --start="Once upon a time" \
    --max_new_tokens=200 --temperature=0.8 --top_k=50

# 导出成 llama2.c 格式（v0 float32 / v1 float32+header / v2 int8 量化）
python export.py outmini/model_v2.bin --version=2 --checkpoint=outmini/ckpt.pt
```

## 5. 测试

```bash
pytest tests/smoke_test.py -v
```

覆盖内容（全部 CPU、合成数据、约 5 秒跑完）：

- RoPE / 因果 mask / KV cache 三条路径的**数值等价性**（增量解码结果 == 整段前向）；
- 端到端训练 4 步并写 `ckpt.pt` + `model.bin`；
- 续训：带 `_orig_mod.` 前缀（`--compile` 产物）的 checkpoint 能否正确恢复权重；
- PPO `step()` 能跑通且统计量有限；
- `sample.py` 命令行能真正生成文本。

## 6. 已知限制 / 后续可做

见 [`OPTIMIZATION.md`](OPTIMIZATION.md)（本轮代码审计 + 已修复项 + 待办清单）。
简要版：

- `train.py --ppo` 未接真实 RL 循环（最大的一块功能缺口）；
- DDP 下只有 rank0 做评估，其他 rank 会“超前”，严格说需要 barrier 或全 rank 评估；
- `compute_advantages` 里的 GAE 是 Python 逐 token 循环（可向量化，但要小心数值稳定性）；
- 暂无 wandb/tensorboard 接入、暂无学习率/梯度监控面板。
