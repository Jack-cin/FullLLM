# 代码审计与优化报告

> 两轮工作的结论：
> **第 1 轮**——仓库原本跑不起来（3 个致命缺陷）且存在一批静默正确性与性能问题；
> **第 2 轮**——把 `ppo.py` 真正接进训练循环（此前 `--ppo` 只是换了个模型头、损失仍是交叉熵），
> 并修掉接通过程中暴露出的 5 个链路级 bug。
>
> 现在：`pytest tests/smoke_test.py` **18 项全过**（纯 CPU、合成数据、约 6 秒），
> 且用仓库自带的 `data/ckpt.pt`（iter 98000 的真实 TinyStories 预训练模型）
> 完成了「PPO 训练 → 采样 → int8 导出」的端到端实测。

验证环境：无 GPU、2 核 CPU、无 TinyStories 原始数据集（测试用合成 token 流）。

---

## 一、致命问题（P0：不修就跑不起来 / 静默失效）

### 第 1 轮已修复

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| 1 | `train.py` `__init__` | 调用 `self._build_dataloader()`，定义却是 `_build_dloader` → **任何一次启动都 `AttributeError`**（实测复现） | 改名并保留别名 |
| 2 | `train.py` 存 ckpt | `asdict(argparse.Namespace)` → 第一次评估保存时 `TypeError`，**训练白跑** | 新增 `config_to_dict()` |
| 3 | `train.py` 续训 | ckpt 键带 `_orig_mod.`（`--compile` 产物）却不剥前缀 → `strict=False` **静默丢弃 100% 权重**（实测 21/21 张量全 missing） | 新增 `load_model_state()`，缺失过半直接报错 |
| 4 | `train.py` resume 路径 | 写死找 `out_dir/ckpt.pt`，而 PPO 存的是 `ppo_ckpt.pt` → `FileNotFoundError` | 新增 `--ckpt_path` + 自动查找 |
| 5 | `train.py` DDP 分支 | 一句 `exit()` → `torchrun` 多卡启动立刻退出 | 删除 |
| 6 | `sample.py` | 依赖不存在的 `configurator.py`；读 `ckpt["config"]` 会 KeyError | 重写为 argparse CLI |
| 7 | `requirements.txt` | 缺 `tyro`（`ppo.py` 顶部 import 它）；版本全锁死在 py3.11 装不上 | 补齐并放宽 |

### 第 2 轮已修复（接通 PPO 时暴露）

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| 8 | `train.py --ppo` | **名不副实的 PPO**：只把模型换成 `TransformerWithValueHead`、ckpt 改名 `ppo_ckpt.pt`，损失仍是交叉熵；`ppo.py` 的 `PPOTrainer` 全仓库**没有任何调用点** | 新增 `_build_ppo()`（策略 + 冻结 `π_ref` + 奖励函数 + `PPOTrainer`）与 `train_ppo()`（rollout → 奖励 → GAE → clip 更新 → 存档） |
| 9 | `model.py` `generate_withcache` | **在整条序列里找 eos**：prompt 通常以 bos 开头（本项目 bos=1、eos=2，`sample.py` 还错传 `bos_id` 当 eos_id）→ 命中的是 prompt 第一个 token，`toks[:1]` 把生成结果整段吞掉，**打印出来是空白** | 只在「新生成的那一段」里找 eos；`sample.py` 改传 `eos_id` |
| 10 | `model.py` 同上 | `echo=True` 时又拼一次 prompt（**prompt 被打印两遍**） | 语义修正：`echo=True` = prompt+生成，`echo=False` = 纯生成 |
| 11 | `train.py` ckpt 元信息 | 续训时结构参数只覆盖了局部字典、没写回 `arg` → 存出的 ckpt 里 `model_args.dim=288`（默认值）**而权重是 dim=64**，`sample.py`/`export.py` 按 config 重建模型必然 size mismatch，**训练→采样闭环断掉** | 结构参数回写 `arg`；新增 `checkpoint_args()`，保存时以模型真实结构为准 |

## 二、正确性 / 静默错误（P1）

### 第 1 轮已修复

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| 12 | `ppo.py` `__init__` | `backward_batch_size / global_batch_size` 三个运行时字段**全仓库无人计算**，`step()` 里 `range(0, bs, None)` 必 TypeError | 构造时按 `mini_batch_size × grad_accum` 补算 |
| 13 | `ppo.py` 自适应 KL | 访问 `config.horizon`，而 `PPOConfig` 没有该字段 → AttributeError | 新增 `horizon` 字段 |
| 14 | `ppo.py` `__init__` | `np.random.seed(config.seed)` 写在 `config or PPOConfig()` 之前 → 不传 config 就崩 | 调整顺序 |
| 15 | `ppo.py` 统计聚合 | `stack_dicts` 用 `-1` 填充，形状不齐时 **-1 被当成真实值参与均值**，日志里的 loss/kl 全错 | NaN 填充 + `nanmean` |
| 16 | `export.py` | `ModelArgs(**ckpt['model_args'])` 直接灌入含训练超参的 config → `TypeError: unexpected keyword argument 'batch_size'` | 只挑模型结构字段 |
| 17 | `train.py` CLI | `--dtype` 的 choices 漏了 `float32`（代码里其实支持） | 补上 |
| 18 | `tinystories.py` | CPU 训练也强制 `pin_memory=True` | 按 device 判断 |

### 第 2 轮已修复

| # | 位置 | 现象 | 修复 |
| --- | --- | --- | --- |
| 19 | `train.py` 学习率 | `--decay_lr` 默认 True → **`--ppo_lr` 被余弦调度静默覆盖**（实测 `ppo_lr=1e-4` 实际只用 1e-5）；resume 场景（`iter_num` 很大）还会直接衰减到 `min_lr=0` ⇒ **PPO 全程不更新参数**，比报错更隐蔽 | `--decay_lr` 默认改为按阶段（预训练 True / PPO False）；`get_lr` 支持显式衰减终点；lr=0 时显式告警 |
| 20 | `train.py` PPO 步数 | `iter_num ≥ max_iters` 时 **PPO 一步都不跑就退出**（接着 iter 98000 的 ckpt 做 2000 步 PPO，而 `--max_iters=2000` → 直接结束） | PPO 阶段自动把 `--max_iters` 解释为「本阶段再训多少步」并打印；新增 `--iters_from_now` 显式声明 |
| 21 | `train.py` DDP | 忽略名单写的是 `freqs_cis`（llama 官方仓的名字），本模型实际叫 `freqs_cos/freqs_sin` → 这个「忽略」一直是空操作，DDP 每步前向都会尝试广播恒定 buffer | 改成真实名字 |
| 22 | `train.py` 存档 | 存的是 `raw_model.state_dict()`，`--compile` 时会写入 `_orig_mod.` 前缀（就是 P0-3 的根因） | 新增 `self.unoptimized_model` 引用，存档/导出/PPO 参考模型都用它 |
| 23 | `ppo.py` 参考模型 | `π_ref` 带 KV cache buffer 却只做整段前向 → 白占一倍显存 | 构建时释放 `cache_k/cache_v` |

## 三、性能优化（P2，均验证过数值等价）

### 第 1 轮

| # | 优化 | 实测 |
| --- | --- | --- |
| 24 | **数据迭代器长生命周期复用** | `iter_batches()` 每次调用都新建 DataLoader（`num_workers>0` 时重新 fork 子进程），而原代码**每个梯度累积 micro-step 都调用一次**。同一 6 步训练（`grad_accum=4, num_workers=2`）：**创建 89 次 → 4 次**，单步 **100~312ms → 34~112ms** |
| 25 | **生成时只投影最后一个位置** | V=32000、T=512：**359.9ms → 0.78ms**（约 460×） |
| 26 | **因果注意力免 mask**（`is_causal=True`） | 省掉每次前向构造 `[T,T]` 的 `-inf` mask（T=512 时 0.8ms），GPU 上还能走 FlashAttention 因果内核；与显式 mask 的 SDPA 结果**逐元素一致** |
| 27 | **KV cache 路径等价性回归** | 逐 token 增量解码 == 整段前向（logits 全等，atol 1e-5）；cache 生成 == 逐步重算的贪心解码 |
| 28 | RoPE 去掉 dtype 往返 | 仅在 fp16 下升 fp32；算子 15.2ms → 14.4ms（细节见代码注释） |
| 29 | DDP 梯度累积改用 `no_sync()`；补 `set_float32_matmul_precision("high")`；`--num_workers` 可配；KV cache 不再每步 `.to()` | 训练吞吐 A/B（loss 完全一致 6.1159）无回归 |

### 第 2 轮

| # | 优化 | 实测 |
| --- | --- | --- |
| 30 | **PPO 的质量监控用小批量数据流**（新增 `--ppo_eval_batch_size`，默认 16） | 用训练 `batch_size=128` 时一次评估前向要 ~3 秒（CPU）→ 换成 `batch_size=8` 后**整轮评估 ~1.6 秒内**（实测每步 1.4~2.8s 含 rollout+更新+评估） |
| 31 | **PPO 质量监控本身**（新增） | 每 `eval_interval` 步用小批量算一次 train/val CE 损失 —— 「奖励涨了但模型不会说人话」的第一预警；无数据/出错时只提示一次、不中断训练 |
| 32 | `--ppo_debug` 样例打印 | 每个日志间隔打印一条真实生成文本 + 奖励，RL 效果肉眼可见 |

## 四、性能归因案例：一次「30 秒卡顿」的排查（结论：不是 bug）

接通 PPO 后我观测到：`eval_interval` 触发的那一步耗时 **33.5 秒**，而正常步只要 ~600ms。
排查过程与结论值得记录，避免后来人误判：

1. `faulthandler` 抓栈 → 卡在 `scaled_dot_product_attention`；独立脚本同算子只要 0.7~3ms ⇒ 不是算子问题。
2. 逐层打点 → **每次评估前向恒定 ~3000ms**，而 `rollout`（含 48 步增量生成）只要 ~300ms。
3. `torch.profiler` → `_softmax` 231ms/次、`bmm` 109ms/次，**所有算子慢 ~100 倍** ⇒ 像是环境级问题。
4. 在 trainer 初始化各阶段后做对照 bench → `_build_model/_build_optimizer/_build_ppo` 之后都是 ~25ms ⇒ 与 PPO 无关。
5. 打印实际输入形状 → **评估批次是 `(128, 256)`**（`train.py` 默认 `batch_size=128 / max_seq_len=256`），
   而我的对照 bench 用的是 `(16, 48)`：**规模差 42 倍以上，再叠加注意力的 T² 项**，3 秒完全对得上。

⇒ **没有 bug**，是「用训练批量做评估」的代价。已按此结论把 PPO 的监控评估改成小批量（见 P2-30）；
普通训练如果要在 CPU 上跑，建议同时调小 `--eval_iters` / `--batch_size`（GPU 上无感）。

## 五、PPO 现在怎么用（实测数据）

用仓库自带的 `data/ckpt.pt`（真实 TinyStories 小模型，iter 98000）+ 10 条英文故事开头做演示：

```bash
python train.py --out_dir=/tmp/ppo --device=cpu --dtype=float32 \\
    --vocab_source=custom --vocab_size=512 --tokenizer_path=data/tok512.model \\
    --init_from=resume --ckpt_path=data/ckpt.pt --ppo \\
    --ppo_prompt_file=prompts.txt --start_len=16 --target_len=48 --max_gen_len=48 \\
    --ppo_rollout_size=16 --ppo_mini_batch_size=8 --ppo_epochs=2 \\
    --ppo_lr=5e-5 --ppo_reward=repetition --init_kl_coef=0.2 \\
    --max_iters=30 --eval_interval=30 --log_interval=10
```

实测输出（节选）：

```
📣 PPO 配置：rollout=16 × 2 epoch（mini-batch=8）、prompt=16 tok、生成=48 tok、
             奖励=repetition（固定目标长度 48）、KL 系数 0.2、lr 5e-05
[PPO] step 98000 | 奖励 +0.978 (本批 +0.978 ± 0.034) | KL_ref +0.0000 | 熵 1.403 | clipfrac 0.021 | lr 5.00e-05 | 633 ms
       └─ 样例（奖励 +1.000）：… The magic snake was very happy and smiled. It was green and scary
📊 CE 损失（质量监控）：train 17.98 | val 17.92
💾 已保存 /tmp/ppo/ppo_ckpt.pt（含 KL 系数 0.2000）
```

奖励选择上的实测体会（写进 README 了）：

- `repetition`（n-gram 不重复率）是密集、可达的信号：**本批唯一率 0.973 → 0.995**，训练 20 步就能看到变化；
- `completion`（结尾标点）对 512 词表小模型**过于稀疏**：实测奖励长期恒为 0，`pg_loss` 无有效梯度，还容易把 `KL_ref` 推到 9~16（漂移失控）；
- `length` 适合控篇幅（本仓库 `LengthReward`），但要注意「奖励黑客」——长度奖励可以被灌水文本骗到，
  所以务必同时盯 CE 损失 + 生成样例。

闭环验证：`ppo_ckpt.pt` 可直接被 `sample.py` 加载（缺失 0 / 多余 2 = value_head，符合预期，生成 353 tok/s），
也可用 `export.py --version=2` 导出 int8（max group error 0.0072）。

## 六、还没做（P3，按收益排序）

1. **DDP 的评估/存档同步**：目前只有 rank0 做评估与存档，其余 rank 靠 all-reduce 隐式同步，长评估间隔下可能看到 NCCL 等待。
2. **GAE 向量化**：`compute_advantages` 是 T 次 Python 循环；可向量化，但要在 fp32/log 空间做以免 `(γλ)^T` 下溢。
3. **奖励模型接入**：现在只有启发式奖励（长度/结尾/重复度）。接口已留好（`__call__(tokens) -> float`）。
4. **可观测性**：指标只打 stdout，接 TensorBoard/wandb 会更利于调参（PPO 尤甚）。
5. **CI / 工程化**：GitHub Actions 跑 `pytest`；`data/*.pt` 这类二进制资产移到 release。
6. **DDP 下的 PPO**：`train_ppo` 已按数据并行写法组织（各 rank 采样、梯度由 trainer 内部 all-reduce），但**没有多卡环境实测**。

## 七、验证方式

```bash
pip install -r requirements.txt
pytest tests/smoke_test.py -v      # 18 passed（CPU，合成数据，约 6 秒）
```

覆盖点（每个测试的注释都写明「修复前会怎么坏」）：

- 工具函数：`asdict(Namespace)` 确实抛 TypeError；ckpt 路径解析（`ppo_ckpt.pt` 优先）；带 `_orig_mod.` 前缀的权重能真正加载（扰动后校验）；
- 模型：RoPE 三重循环参照实现对照（fp32/bf16）；`is_causal` SDPA == 显式 mask；KV cache 增量解码 == 整段前向；cache 生成 == 贪心参照；
- **生成语义回归**：prompt 以 eos id 开头时输出不被吞掉；`echo` 语义（含/不含 prompt）；
- 端到端训练：合成 bin 数据 → 4 步训练 → 校验 `ckpt.pt` / `model.bin` 产出；
- 续训：带前缀 ckpt 恢复权重；PPO 阶段优先 `ppo_ckpt.pt`；
- **PPO**：奖励函数三种模式与别名解析；`_rollout` 形状约束（等长 prompt → 无 padding）；`train_ppo` 端到端产出含 `kl_coef` 的 ckpt、权重名无 `_orig_mod.`；续训恢复 KL 系数；
- `sample.py` CLI 端到端生成。
