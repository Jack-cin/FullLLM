# 代码审计与优化报告

> 结论先说：**有优化空间，而且最大的空间不在超参，而在代码本身**——
> 本轮审计发现 3 个「让程序根本跑不起来 / 静默失效」的致命问题、若干正确性与性能问题。
> 致命问题与一批低风险性能优化已在本次改动中修复，并用 13 项冒烟测试（纯 CPU、合成数据）验证通过。

验证方式：本仓库所有测试均在无 GPU、无 TinyStories 原始数据集的条件下运行
（`pip install -r requirements.txt && pytest tests/smoke_test.py -v`）。

---

## 一、致命问题（P0：不修就是白跑/跑不起来）——已修复

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| 1 | `train.py` `TinyStoriesTrainer.__init__` | 调用 `self._build_dataloader()`，但定义的方法名是 `_build_dloader` → **任何一次启动都 `AttributeError`**，训练脚本完全无法运行（实测：`AttributeError: 'TinyStoriesTrainer' object has no attribute '_build_dataloader'`） | 方法重命名为 `_build_dataloader`，保留 `_build_dloader` 别名 |
| 2 | `train.py` 保存 checkpoint | `"model_args": asdict(arg)`，而 `arg` 是 `argparse.Namespace` 不是 dataclass → 第一次评估保存时 `TypeError: asdict() should be called on dataclass instances`，**训练全部白跑** | 新增 `config_to_dict()`，Namespace / dataclass 双分支 |
| 3 | `train.py` 续训加载权重 | checkpoint 里的 key 带 `_orig_mod.` 前缀（`--compile` 训练的产物），加载时却不剥前缀 → `strict=False` **静默丢弃 100% 权重**（实测 21/21 个参数张量全部 missing），所谓「PPO 微调」其实是拿随机权重在训 | 新增 `load_model_state()`：剥前缀 + 缺失项显式打印 + 缺失过半直接报错 |
| 4 | `train.py` resume 路径 | 写死找 `out_dir/ckpt.pt`，但 PPO 阶段存的是 `ppo_ckpt.pt`，且 README 里两阶段的 `out_dir` 不同 → `FileNotFoundError` | 新增 `resolve_ckpt_path()`：`--ckpt_path` > `ppo_ckpt.pt` > `ckpt.pt`，模型与优化器共用同一份解析结果 |
| 5 | `train.py` DDP 分支 | `print("self.ddp_world_size", ...)` 之后紧跟一句 `exit()` → 只要用 `torchrun` 启动就立刻退出，**多卡训练完全不可用** | 删除 `exit()`（它本身也只存在于交互解释器） |
| 6 | `sample.py` | `exec(open('configurator.py').read())`，仓库里没有这个文件（nanoGPT 残留）→ `FileNotFoundError`；`checkpoint_dict["config"]` 在自训 ckpt 上会 `KeyError` | 全量改写为 argparse CLI；兼容 `model_args`/`config` 两种 ckpt；剥离 `_orig_mod.` 前缀 |
| 7 | `requirements.txt` | 缺 `tyro`（`ppo.py` 顶部 `import tyro`）→ `ppo.py` 直接 ImportError；版本全量 `==` 锁死在 py3.11+ 装不上 | 补齐依赖并把 `==` 改为 `>=`（`numpy` 仍锁 `<2`，因为 torch 2.0 不兼容 numpy 2.x） |

## 二、正确性 / 静默错误（P1）——已修复

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| 8 | `ppo.py` `PPOTrainer.__init__` | `backward_batch_size / global_batch_size` 三个运行时字段**全仓库无人计算**，`step()` 里 `range(0, bs, None)` 必然 TypeError | 构造时按 `mini_batch_size × grad_accum` 补算 |
| 9 | `ppo.py` 自适应 KL | `adap_kl_ctrl=True` 时访问 `config.horizon`，而 `PPOConfig` 没有该字段 → AttributeError | 新增 `horizon` 字段（默认 10000，对齐 OpenAI 实现） |
| 10 | `ppo.py` `__init__` | `np.random.seed(config.seed)` 写在 `self.config = config or PPOConfig()` 之前 → 不传 config 就崩 | 调整顺序 |
| 11 | `ppo.py` 统计量聚合 | `stack_dicts` 用 `padding_value=-1` 填充，mini-batch 形状不齐时 **-1 被当成真实值参与均值**，日志里的 loss/kl 全是错的（甚至出现负 loss） | NaN 填充 + `nanmean`，并修掉 `[0]` 取值的截断 |
| 12 | `export.py` `load_checkpoint` | `ModelArgs(**checkpoint_dict['model_args'])`，而 train.py 存的是**整个训练配置**（含 `batch_size` 等）→ `TypeError: unexpected keyword argument 'batch_size'` | 只挑模型结构字段 + 兼容 `config` 键 |
| 13 | `train.py` CLI | `--dtype` 的 choices 只有 `float16/bfloat16`，但精度映射表支持 `float32` → CPU 训练想指定 float32 会被 argparse 拒绝 | choices 补上 `float32` |
| 14 | `tinystories.py` `iter_batches` | `pin_memory=True` 恒定开启，CPU 训练时白占锁页内存并多一次拷贝 | 按 `device` 判断；worker>0 时开启 `persistent_workers` + `prefetch_factor` |

## 三、性能优化（P2）——已验证等价、已应用

| # | 优化 | 依据 / 实测 |
| --- | --- | --- |
| 15 | **因果注意力免 mask**：`start_pos==0` 且不走 KV cache 时传 `is_causal=True`、不再构造 `[T,T]` 的 `-inf` mask | 实测 SDPA 结果与显式 mask **逐元素一致**（`tests/smoke_test.py::test_sdpa_is_causal_matches_explicit_mask`）；省掉每次前向 0.8 ms 的 mask 构造（T=512）。CPU 上两种路径耗时相当，**收益主要在 GPU**：FlashAttention 的因果内核不再需要物化 `[B,H,T,T]` 加性 mask，长序列下省显存、避免额外 mask 读写 |
| 16 | **增量解码 == 整段前向**（回归测试，防止上面改动破坏语义） | `test_kv_cache_decode_matches_full_forward`：逐 token 带 cache 前向 vs 整段前向，logits 全等（atol 1e-5）；`test_generate_withcache_matches_greedy_reference`：KVCache 生成与逐步重算的贪心解码结果完全一致 |
| 17 | **生成时只投影最后一个位置** | 原代码对整段 prompt 都做 vocab 投影。实测（T=512, V=32000, fp32/CPU）：**359.9 ms → 0.78 ms**，约 460×；prompt 越长越明显 |
| 18 | **RoPE 去掉 dtype 往返** | 原实现无条件 `x.float()` + `.type_as(xq)`，每层每个 head 两次转换。改为仅在 fp16 下升 fp32 计算；实测 RoPE 算子 15.2 ms → 14.4 ms（fp32，B8/T512/H8/D64），GPU 上省的是 kernel 启动与显存读写。数学等价（freqs 本身 fp32，bf16 会按类型提升到 fp32） |
| 19 | **KV cache 不再每步 `.to()`** | 只在 device/dtype 真的不一致时搬一次（对 `torch.compile` 更友好） |
| 20 | **DDP 梯度累积改用 `no_sync()`** | 原来直接改私有属性 `model.require_backward_grad_sync`，在 compile / 新版 torch 下随时可能失效或静默不生效 |
| 21 | **打开 TF32 的新式开关** | 补 `torch.set_float32_matmul_precision("high")`，与 `allow_tf32` 一起才算完整打开 |
| 22 | **DataLoader 并行度可配** | 新增 `--num_workers`（原为硬编码 0，读盘与训练抢主线程） |
| 23 | **训练吞吐 A/B 实测** | 同一脚本在「修复前/后」的 `model.py` 上跑 fwd+bwd（dim64/5层/T512/B32，CPU 4 线程）：3048 vs 3043 ms、3348 vs 3217 ms，**loss 完全一致（6.1159）** → 无回归，CPU 侧持平，收益集中在 GPU 与推理路径 |
| 24 | **数据迭代器长生命周期复用**（本轮收益最大的一项性能优化） | `Task.iter_batches()` 每调用一次就新建一个 `DataLoader`，`num_workers>0` 时还要**重新 fork 子进程**；而原代码在**每个梯度累积 micro-step** 都 `next(iter_batches(...))` 一次。实测同一个 6 步训练（`grad_accum=4`、`num_workers=2`）：**修复前创建 89 次 DataLoader → 修复后 4 次**（train/val 各 2 个 worker）；单步耗时 100~312 ms → 34~112 ms。正确性不受影响（每个 split 只做一次 `__iter__`，内部本来就是无限流） |

## 四、还没做 / 需要你拍板的（P3 及以后）

按「收益 ÷ 工作量」排序：

1. **把 `ppo.py` 真正接进训练循环（最大功能缺口，工作量 1~2 天）**
   现状：`--ppo` 只换了模型头，损失仍是交叉熵；`PPOTrainer` 是一份可用的库（`step()` 已验证能跑）。
   要补的是 rollout 循环：批量取 prompt → `generate_withcache(..., temperature=0.8)` → 用 `LengthReward`
   （或自定义奖励模型）打分 → `PPOTrainer.step()`。同时建议：奖励尺度（`LengthReward` 除以 100）、
   `init_kl_coef`、`cliprange`、`vf_coef` 需要一起调；`entropy` 目前只统计不参与损失，可加熵奖励系数。
2. **DDP 的评估/日志同步（半天）**
   现在只有 `master_process` 做 `estimate_loss`，其余 rank 会继续跑，靠 backward 的 all-reduce 兜住，
   严格说属于「隐式同步」，长评估间隔下容易看到 NCCL 等待/超时。
3. **GAE 向量化（半天）**
   `compute_advantages` 是 T 次 Python 循环（T=响应长度）。可用 `(γλ)` 幂次 + 翻转 cumsum 向量化，
   但 fp16 下要按 log 空间或 fp32 计算，避免 `(γλ)^T` 下溢；建议保留当前实现作为参照测试。
4. **量化导出与推理（半天）**
   `export.py` 已有 v2（int8 Q8_0）导出，但 `train.py` 每次评估都调 `model_export(..., version=0)`
   写 fp32 bin。可加 `--export_version`，或只在训练结束时导出一次。
5. **可观测性（半天）**
   没有 wandb/tensorboard；`omni` 的 loss/lr/耗时只打到 stdout。加 tensorboard 或 csv 日志后，
   调参效率会有质变（尤其是 PPO 阶段看 `ppo/kl_ref`、`ppo/reward_all` 曲线）。
6. **数据侧（1 天）**
   `PretokDataset` 每个 shard 都 `np.memmap`，多 worker 下内存映射数会随 worker 数放大；
   预分词阶段已经用了进程池，但 `pretokenize` 目前无进度明细（`ProcessPoolExecutor.map` + `tqdm` 可解）。
7. **工程化（按需）**
   CI（GitHub Actions 跑 `pytest`）、`pre-commit`、把 `data/*.pt` 这类二进制资产移到 release / 对象存储。

## 五、本次改动的验证方式

```bash
pip install -r requirements.txt
pytest tests/smoke_test.py -v      # 13 passed（CPU，合成数据，约 5 秒）
```

- 每个测试的注释都写清了「修复前会怎么坏」，例如
  `test_config_to_dict_handles_namespace` 里显式断言 `asdict(Namespace)` 会抛 `TypeError`；
  `test_load_model_state_strips_compile_prefix` 会扰动权重后校验是否真的加载成功。
  （注：`estimate_loss` 的签名随之从 `iter_batches` 工厂改为 `next_batch(split)`，仓库内只有一处调用点，已同步。）
- 端到端：`test_training_smoke_writes_checkpoint_and_export` 用**合成 token 流**造出
  `data/tok512/*.bin`，跑 4 步训练并校验 `ckpt.pt` / `model.bin` 真实产出；
  `test_sample_cli_runs` 用仓库自带的 `data/ckpt.pt` 走完整生成流程。
