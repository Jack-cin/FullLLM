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

### 3.1 普通续训

```bash
python train.py --init_from=resume --out_dir="outmini" --device=cuda \
    --learning_rate=1e-4 --max_iters=200000 --always_save_checkpoint
```

续训会自动在 `out_dir` 里按 `ppo_ckpt.pt → ckpt.pt` 查找（也可用 `--ckpt_path=...` 显式指定）；
若 ckpt 里的 `iter_num` 已经 ≥ `--max_iters`（例如接着 iter 98000 的模型继续训），
`--max_iters` 会被解释成「再训多少步」并打印提示，也可以用 `--iters_from_now` 显式声明该语义。

### 3.2 真正的 PPO 微调（`--ppo`）

`--ppo` 会启用 `ppo.py` 里的 `PPOTrainer`，跑完整的强化学习循环：
**取 prompt → 策略模型采样 response → 奖励打分 → 参考模型算 KL → GAE → PPO-clip 更新**。

```bash
# 从预训练 ckpt 出发做 PPO（本仓库自带 data/ckpt.pt，可直接拿来试）
python train.py --out_dir=out_ppo --device=cuda --dtype=bfloat16 \
    --vocab_source=custom --vocab_size=512 --tokenizer_path=data/tok512.model \
    --init_from=resume --ckpt_path=data/ckpt.pt \
    --ppo \
    --ppo_prompt_file=prompts.txt \        # 每行一条故事开头（留空则从训练集取真实故事开头）
    --start_len=16 --target_len=48 --max_gen_len=48 \
    --ppo_rollout_size=32 --ppo_mini_batch_size=8 --ppo_epochs=2 \
    --ppo_lr=1e-4 --ppo_reward=repetition,length --ppo_temperature=0.9 \
    --init_kl_coef=0.2 --target_kl=0.1 --ppo_early_stopping \
    --max_iters=2000 --eval_interval=100 --log_interval=10
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--ppo_reward` | 奖励函数，可逗号组合：`length`（长度对齐目标）/ `completion`（结尾是句末标点）/ `repetition`（n-gram 不重复率，专治小模型复读）；也支持 `both`、`all` |
| `--ppo_prompt_file` | prompt 文本文件（每行一条）；不给则用训练集分片里的真实故事开头 |
| `--ppo_rollout_size` | 每个 PPO step 采样多少条 (prompt, response)，不能超过 `batch_size` |
| `--ppo_mini_batch_size` | 更新时的 mini-batch（默认 `rollout/8`） |
| `--ppo_epochs` | 同一批 rollout 复用几轮 |
| `--ppo_lr` | PPO 阶段学习率（0 = 沿用 `--learning_rate`）。PPO 阶段默认**不做**余弦衰减 |
| `--init_kl_coef` / `--target_kl` | KL 惩罚系数 / 早停阈值（`--ppo_early_stopping` 时生效） |
| `--ppo_kl_penalty` | `kl` / `abs` / `mse` / `full` |
| `--ppo_eval_batch_size` | 质量监控（CE 损失）用的小批量，默认 16 |

训练日志每 `--log_interval` 步打印一行，包含：
奖励、`KL_ref`（策略相对参考模型的漂移）、`policykl`、熵、`clipfrac`（被裁剪的比率占比）、损失、平均响应长度。
另外每 `--eval_interval` 步会用小批量算一次**验证集 CE 损失**——它是「奖励涨了但模型不会说人话」这类事故的第一预警。

**调参建议**（都基于本仓库实测）：

- 奖励从 `repetition` 或 `repetition,length` 起步：它是密集信号、马上能看出效果；
  `completion` 对 512 词表的小模型过于稀疏（实测奖励长期恒为 0，几乎学不动）。
- `--ppo_lr` 取预训练的 1/10 ~ 1/25（`5e-5 ~ 2e-4`），`--init_kl_coef 0.1~0.2`；
  `KL_ref` 持续 > 10 说明漂移太大，调小 lr 或加大 KL 系数。
- rollout 越大梯度越稳（`32~128`），但每步耗时线性增长；先用 `--ppo_rollout_size=16` 试通。
- 奖励/长度都是「可被 hack 的目标」，所以训练中一定要同时看 CE 损失与生成样例（`--ppo_debug` 默认开启样例打印）。

> 想换成自己的奖励（reward model / 人工偏好）：实现一个 `__call__(tokens) -> float` 的对象，
> 在 `_build_ppo` 里替换 `self.reward_fn` 即可（见 `train.py::TinyStoriesReward`）。

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


- DDP 下只有 rank0 做评估/存档，其他 rank 会“超前”（梯度同步靠 all-reduce 兜住）；
- `compute_advantages` 里的 GAE 是 Python 逐 token 循环（可向量化，但要小心数值稳定性）；
- 暂无 wandb/tensorboard 接入（指标目前打 stdout，便于接任意 logger）。
