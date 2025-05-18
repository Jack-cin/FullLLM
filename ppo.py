'''
1. 初始化阶段
定义一些基本参数并验证输入数据的一致性。对于每个训练样本，将查询 q_i 和响应 r_i 拼接成完整的输入 x_i = [q_i; r_i]
数学公式表示为： x_i = [q_i; r_i] ∈ ℕ^{n_i + m_i}
其中 q_i 和 r_i 分别为查询和响应，长度分别为 n_i 和 m_i，拼接后的 x_i 为输入数据。

2. 数据预处理
将查询和响应通过 torch.cat 拼接，并生成注意力掩码 attention_mask，其中每个有效的token标记为1。
数学公式表示为： M_i ∈ {0,1}^{n_i + m_i}
填充后的序列通过 tokenizer 进行处理，确保所有输入都填充至相同长度。

3. 策略评估阶段
在策略评估阶段，我们通过前向传播计算当前策略和参考模型的log概率（logprobs）和价值函数（values）。
然后计算KL散度，衡量当前策略和参考模型之间的差异。
KL散度公式为： D_{KL}(\pi_{\theta} || \pi_{\text{old}}) = \frac{1}{B} \sum_{i=1}^{B} \sum_{t=1}^{m_i} (log \pi_{\theta}(r_t^i) - log \pi_{\text{old}}(r_t^i)) * M_t^i
其中 B 是批次大小，m_i 是第 i 个样本的序列长度，M_t^i 是对应token的有效掩码。

4. 奖励计算阶段
奖励 r_i 计算为包含KL惩罚的综合奖励。KL惩罚项根据策略之间的差异来计算。
奖励计算公式为： r_i = R_{\text{RM}}(q_i, r_i) - \beta * \sum_t (log \pi_{\theta}(r_t | q_i) - log \pi_{\text{old}}(r_t | q_i)) * M_t
其中 β 是KL惩罚系数。

5. 优势函数与回报计算阶段
优势函数 A_t^{GAE} 使用广义优势估计（GAE）方法计算。其公式为：
A_t^{GAE} = ∑_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}
其中 δ_t = r_t + \gamma V(s_{t+1}) - V(s_t) 为TD残差，γ 为折扣因子，λ 为平滑因子。
回报 R_t 为优势函数与值函数的和： R_t = A_t^{GAE} + V(s_t)

6. 优化阶段
在优化阶段，PPO算法通过损失函数优化模型。损失函数包括策略损失（Clip Loss），值函数损失（Value Loss），以及熵奖励（Entropy），其组合为：
L = L_{\text{Clip}} + c_vf L_{\text{VF}} - c_e L_{\text{Ent}}
其中：
策略损失 L_{\text{Clip}} 为： L_{\text{Clip}} = - \mathbb{E} [ \min( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t ) ]
值函数损失 L_{\text{VF}} 为： L_{\text{VF}} = \mathbb{E} [(V_{\theta} - G)^2]
熵损失 L_{\text{Ent}} 为： L_{\text{Ent}} = \mathbb{E} [ -\sum_a \pi_{\theta}(a|s) \log \pi_{\theta}(a|s) ]

7. 提前停止
如果KL散度超过设定的阈值，则触发提前停止机制。提前停止的条件是：
D_{KL}(\pi_{\theta} || \pi_{\text{old}}) > 1.5 * target_kl
当满足该条件时，训练提前终止。
'''




from dataclasses import dataclass, field
from typing import Literal, Optional, List

import tyro
# -------------------- 基础计算库 --------------------
import math  # 数学运算（如cosine调度器计算）
import os   # 系统路径操作（模型保存路径管理）
import time  # 训练计时与性能分析
import warnings # 警告抑制（过滤CUDA警告等）

# 数值计算核心库（与PyTorch张量交互使用）
import numpy as np

# -------------------- 深度学习框架 --------------------
import torch # 张量计算与自动微分（版本要求2.0+）
from torch import nn  # 神经网络模块（自定义模型继承nn.Module）
import torch.nn.functional as F # 激活函数与损失函数（如F.cross_entropy）

# 序列处理工具（用于变长输入的batch处理）
# 示例：将不同长度的文本序列填充为相同长度
from torch.nn.utils.rnn import pad_sequence

# -------------------- 兼容性处理 --------------------
# 适配不同Python版本的抽象基类导入方式
# 说明：Python 3.3+将Mapping移到collections.abc
try:
  from collections.abc import Mapping # 新版导入方式
except ImportError:
  from collections import Mapping # 旧版本回退

# 配置类：PPOConfig用于管理PPO算法的超参数
'''
1. 初始化配置阶段：
    在这个阶段，PPOConfig 类定义了所有需要的超参数。这些参数包括学习率、训练步数、PPO的剪切范围、奖励计算等。
解释：
    这些配置项是PPO算法的超参数，控制训练的各个方面，例如学习率、训练的步数（steps）、PPO剪切范围（cliprange）等。
    在实际训练过程中，算法会依照这些超参数进行模型训练。
'''
@dataclass
class PPOConfig:
    """
    配置类，用于存储和管理PPO算法的所有参数设置
    """
    seed: int = 0  # 随机种子，确保训练可以重复
    log_with: Optional[Literal["wandb", "tensorboard"]] = None  # 日志记录方式，'wandb' 或 'tensorboard'

    # PPO训练超参数
    steps: int = 20000  # 总训练步数
    learning_rate: float = 1e-5  # 学习率，控制参数更新的步幅
    adap_kl_ctrl: bool = False  # 是否使用自适应KL控制（控制KL散度，调整学习率）
    init_kl_coef: Optional[float] = 0.2  # 初始KL惩罚系数，用于控制KL散度
    kl_penalty: Literal["kl", "abs", "mse", "full"] = "kl"  # KL惩罚计算方式，可以选择'kl', 'abs', 'mse', 'full' #kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution
    target: Optional[float] = 6  # 目标KL值，用于自适应KL控制
    gamma: float = 1  # 用于计算优势（advantage）的折扣因子
    lam: float = 0.95  # 用于计算优势的lambda系数
    cliprange: float = 0.2  # PPO的剪切范围，用于控制策略更新幅度
    cliprange_value: float = 0.2  # 值函数的剪切范围，用于控制价值的更新幅度
    vf_coef: float = 0.1  # 值函数损失的系数
    batch_size: int = 256  # 每次训练批次的样本数量
    mini_batch_size: int = 1  # 每个mini batch的样本数量
    gradient_accumulation_steps: int = 1  # 梯度累积步数，防止显存溢出
    ppo_epochs: int = 4  # 每个batch训练的epoch数
    max_grad_norm: Optional[float] = None  # 梯度的最大范数，用于梯度裁剪
    early_stopping: bool = False  # 是否启用提前停止，若KL散度过大则停止训练
    target_kl: float = 0.1  # 提前停止的KL散度目标值
    ratio_threshold: float = 1.5  # PPO比率的阈值，用于防止梯度爆炸
    whiten_rewards: bool = False  # 是否对奖励进行白化处理（增强训练稳定性）

    # 运行时计算的超参数，初始化时不需要设置
    backward_batch_size: tyro.conf.Suppress[int] = None  # 反向传播中使用的批量大小
    global_backward_batch_size: tyro.conf.Suppress[int] = None  # 跨进程的反向传播批量大小
    global_batch_size: tyro.conf.Suppress[int] = None  # 跨进程的批量大小

# PPO训练器类：用于执行PPO训练过程
class PPOTrainer():
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily based on the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback
    and the HuggingFace implementation trl here: https://github.com/huggingface/trl/tree/main
    """
    def __init__(
        self,
        config: PPOConfig = None,
        model: nn.Module = None,
        ref_model: Optional[nn.Module] = None,
        tokenizer = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        is_ddp: bool = False,
    ):
        # 初始化训练器，设置随机种子和配置
        np.random.seed(config.seed)
        self.config = config or PPOConfig()  # 如果没有传入配置，使用默认配置
        self.model = model  # 使用的语言模型
        self.ref_model = ref_model  # 参考模型，用于计算KL惩罚
        self.tokenizer = tokenizer  # 用于文本的tokenizer
        self.optimizer = optimizer  # 优化器，用于更新模型参数
        self.scaler = scaler  # 用于自动混合精度训练的梯度缩放器
        self.is_ddp = is_ddp  # 是否使用分布式数据并行（DDP）

        # 根据配置选择KL控制方式
        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

    # @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        # === 初始化阶段 ====================================================
        """
        执行PPO优化步骤，计算并返回训练过程中的统计数据
        """
        bs = self.config.batch_size  # 获取训练批次大小
        # 检查查询、响应和得分列表的长度是否一致
        assert len(queries) == len(responses) == len(scores) == bs, "所有列表的长度必须一致"
        timing = dict() # 时间统计器（用于性能分析）
        t0 = time.time() # 时间锚点 t₀
        full_kl_penalty = self.config.kl_penalty == "full"


        # === 数据预处理 ====================================================
        '''
        2. 输入数据准备（张量拼接，拼接查询和响应）：
            每个训练样本由一个查询（query）和一个响应（response）组成，算法将查询和响应拼接在一起作为模型的输入。
        解释： 
            x_i = concat(q_i, r_i)  其中 q_i = [q₁, q₂, …, qₙ], r_i = [r₁, r₂, …, rₘ]
            假设 queries 是问题，responses 是模型的回答。
            通过 torch.cat([q, r]) 将查询和响应拼接在一起，这样每个样本的输入数据就包含了问题和回答的完整信息。
        '''
        # 将查询和响应拼接，形成完整的输入数据
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]

        # 构建注意力掩码（有效token标记），
        input_data = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids] # 拼接后的序列 与 M_i ∈ {0,1}^{n_i+m_i}（初始全1掩码）

        # 填充对齐（生成等长序列）
        model_inputs = self.tokenizer.pad(input_data)  # 使用tokenizer将输入数据填充到最大长度
        model_inputs = {k: v.to(queries[0].device) for k, v in model_inputs.items()}  # 将输入数据移动到合适的设备上
        model_inputs_names = list(model_inputs.keys())

        # === 策略评估阶段 ==================================================
        self.model.eval()  # 冻结策略网络参数 θ
        with torch.no_grad():  # 禁用自动微分
            # 策略网络前向传播（计算当前策略的logprobs（对数概率））
            '''
            3. 前向传播：计算模型的输出（log-probabilities）：
                在这一阶段，我们将拼接后的输入数据送入模型，计算出模型的输出，通常是每个动作（响应）的log-probabilities（对数概率）。
            解释：
                # 数学符号：
                # logπ_θ = Σ_{t=1}^m log π_θ(r_t | q, r_{<t}) （自回归生成的对数概率和）
                # V_θ ∈ ℝ^B （状态价值函数估计Q）
                # masks ∈ {0,1}^{B×T} （有效token位置指示）
            '''
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )

            # 参考模型前向传播（旧策略 π_{old}）
            # 计算参考模型的logprobs（对数概率），用于KL散度的计算
            ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                self.ref_model,
                queries,
                responses,
                model_inputs,
                return_logits=full_kl_penalty,
            )

            # 计算KL散度（度量模型与参考模型的差异）
            '''
            4. 计算KL散度（KL Divergence）：
                参考链接：https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DPPO/
                KL散度是衡量当前策略与参考策略之间差异的一个度量。PPO算法会根据这个散度对模型进行更新。
            解释：
                KL(p_model || p_ref) = E_{r_i} [ log( p_model(r_i|q_i)/p_ref(r_i|q_i) ) ]  
                其中 p_model(r_i|q_i) 是当前模型概率，p_ref(r_i|q_i) 是参考模型概率
                p_model(r_i|q_i)/p_ref(r_i|q_i)：  π/π_old = e^ln(π/π_old) =  e^(lnπ-lnπ_old) = all_logprobs - ref_logprobs
                计算当前模型和参考模型（ref_model）的输出log-probabilities之间的差异，得到KL散度。
                all_logprobs - ref_logprobs 计算的是两者的对数概率差异，masks 用于处理padding，确保只对实际数据计算KL散度。
            '''
            kl_ref = ((all_logprobs - ref_logprobs) * masks).sum(axis=-1).mean()

        timing["time/ppo/forward_pass"] = time.time() - t0

        # ====== 计算奖励(reward) 与优势(advantage) ======
        '''
        1. full _kl_penalty 开关到底差在哪？
            full_kl_penalty = True
                对 查询(query)+回复(response) 的所有 token 加 KL；
                需要对 logits 做一次完整 softmax → log_softmax，内存稍高。
            False
                只对 回复部分 计算 KL（更常见做法），用先前已 gather 的 logprobs 即可。
                节省显存与时间，足以抑制策略漂移。
        2. 整体流程
            前向推断得到 (logits, values) ➜ 先前代码段已完成
            1)计算奖励
                把 偏好分数 注入到回复末 token；
                每 token 加上 −kl_coef × KL，得到 rewards。
            2)GAE
                用折扣 γ 与 trace-decay λ 累积，求 advantages 与 returns。
        '''

        with torch.no_grad():  # ① 只做前向推断与统计，关闭梯度，节省显存与时间
            t = time.time()  # ② 记录“奖励计算”开始时间

            # ---------- ②-1 计算 token 级奖励 ----------
            if full_kl_penalty:  # ③ 若要求对“所有 token”施加 KL 惩罚
                # ④ 将 logits → log-probs，不做 gather；得到形状 [B, T, vocab]
                active_full_logprobs = logprobs_from_logits(
                    logits_or_none, None, gather=False
                )
                # ⑤ 参考模型同理
                ref_full_logprobs = logprobs_from_logits(
                    ref_logits_or_none, None, gather=False
                )

                # ⑥ 综合 RM 分数(score) 与 KL 惩罚，得到每个 token 的最终奖励 r_t
                rewards, non_score_reward = self.compute_rewards(
                    scores,  # 句子级打分（来自奖励模型）
                    active_full_logprobs,  # 当前策略 π 的 logp
                    ref_full_logprobs,  # 参考策略 π_ref 的 logp
                    masks  # 有效 token 掩码
                )
            else:  # ⑦ 仅对 response 段做 KL，已提前 gather
                rewards, non_score_reward = self.compute_rewards(
                    scores,
                    all_logprobs,  # 当前策略在 response 上的 logp
                    ref_logprobs,  # 参考策略在 response 上的 logp
                    masks
                )

            # ⑧ 统计 batch 内平均奖励，用于日志 & 调参观察
            reward_all = (rewards * masks).sum(axis=-1).mean()
            timing["time/ppo/compute_rewards"] = time.time() - t  # ⑨ 记录奖励阶段耗时

            # ---------- ②-2 计算优势 GAE ----------
            t = time.time()  # ⑩ 记录“优势估计”开始时间
            values, advantages, returns = self.compute_advantages(
                values,  # baseline：值函数 V(s_t)
                rewards,  # 刚算好的 token-level 奖励 r_t
                masks  # 有效 token 掩码
            )
            timing["time/ppo/compute_advantages"] = time.time() - t  # ⑪ 记录优势阶段耗时

        # 记录训练数据
        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        # 开始训练
        # ================= PPO 参数更新主循环 =================
        t = time.time()  # ① 记录本轮优化总耗时起点
        all_stats = []  # ② 收集每个 mini-batch 的训练统计
        early_stop = False  # ③ KL 过大时可触发提前终止

        for ep in range(self.config.ppo_epochs):  # ④ 外层：对整个 batch 迭代 ppo_epochs 次
            if early_stop:  # （一个 batch≈一次“PPO epoch”）
                break

            b_inds = np.random.permutation(bs)  # ⑤ 打乱索引，确保每 epoch 随机化

            # ---- 梯度累积（backward_batch）循环 ----
            # backward_batch_size = mini_batch_size * gradient_accumulation_steps
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                # ---- 真正的 mini-batch 循环 ----
                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]

                    # ⑥ 构造当前 mini-batch 的张量 / 列表
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],  # 旧策略 logπ_old
                        "values": batch_dict["values"][mini_batch_inds],  # baseline V(s_t)
                        "masks": batch_dict["masks"][mini_batch_inds],  # 有效 token
                        # queries / responses 是“ragged list”，无法直接切片张量
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],  # ĤA_t
                        "returns": batch_dict["returns"][mini_batch_inds],  # R_t
                    }
                    for k in model_inputs_names:  # ⑦ 补齐 input_ids / attention_mask
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]

                    # ========== 前向 & 计算损失 ==========
                    model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}
                    self.model.train()  # ⑧ 切换到训练模式
                    '''
                    # logprobs:  log π_θ(a_t|s_t) ∈ ℝ^{B×T_resp}
                    # logits:    未归一化分布 z_t ∈ ℝ^{B×T_resp×|V|}
                    # vpreds:    值函数 V_θ(s_t) ∈ ℝ^{B×T_resp}
                    '''
                    logprobs, logits, vpreds, _ = self.batched_forward_pass(  # ⑨ 新策略 π 参数 θ
                        self.model,
                        mini_batch_dict["queries"],
                        mini_batch_dict["responses"],
                        model_inputs,
                        return_logits=True,
                    )

                    '''
                    # 关键公式 (compute_loss 内):
                    #   概率比ratio_t   = exp(logπ_new - logπ_old)              (1)
                    #   策略损失L_clip_t  = min( ratio_t·Â_t ,
                    #                   clip(ratio_t, 1-ε, 1+ε)·Â_t )     (2)
                    #   值函数损失V_clip_t  = ½·max( (V_new-R_t)² ,
                    #                       (clip(V_new, V_old±ε_v)-R_t)² ) (3)
                    #   总损失loss      = -E[L_clip] + c_v·E[V_clip]            (4)
                    #   其中 ε = cliprange, ε_v = cliprange_value
                    #
                    # 张量维度:
                    #   ratio_t, Â_t, L_clip_t → ℝ^{B×T_resp}
                    #   通过 masked_mean() → 标量
                    '''
                    loss, train_stats = self.compute_loss(  # ⑩ 计算 PPO 损失
                        mini_batch_dict["logprobs"],  # π_old
                        mini_batch_dict["values"],  # baseline
                        logprobs, logits, vpreds,  # π_new
                        mini_batch_dict["masks"],
                        mini_batch_dict["advantages"],
                        mini_batch_dict["returns"],
                    )
                    all_stats.append(train_stats)  # ⑪ 收集日志

                    # ⑫ DDP：最后一个 mini-batch 才同步梯度，减少通讯
                    if self.is_ddp:
                        self.model.require_backward_grad_sync = mini_batch_end >= self.config.backward_batch_size

                    # ⑬ 梯度累积：loss 除梯度累积步数
                    loss = loss / self.config.gradient_accumulation_steps # (梯度平均)
                    self.scaler.scale(loss).backward()  # ⑭ AMP 缩放后反向 # dθ ← ∇loss

                # ---- 每个 backward_batch 结束：执行一次 optimizer.step() ----
                if self.config.max_grad_norm is not None:  # ⑮ 梯度裁剪（防梯度爆炸）
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm) # ∥g∥₂ ← clip

                self.scaler.step(self.optimizer)  # ⑯ 更新参数 # θ ← θ - α·ĝ
                self.scaler.update()  # ⑰ 更新缩放器
                self.optimizer.zero_grad(set_to_none=True)  # ⑱ 立即清梯度，释放显存

            # ---- epoch 级早停：监控 KL ----
            if self.config.early_stopping:
                # policykl =  E_t[ logπ_old - logπ_new ]  (近似 KL)
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    print(f"Early stopping at epoch {ep}, policykl is {policykl} > {self.config.target_kl * 1.5}")
                    break

        # ========== 统计与日志 ==========
        timing["time/ppo/optimize_step"] = time.time() - t  # ⑳ 优化总耗时

        t = time.time()
        train_stats = stack_dicts(all_stats)  # ㉑ 把多个 mini-batch 的 dict 堆叠
        stats = {}
        for k, v in train_stats.items():  # ㉒ 取均值写入最终日志键
            stats[f"ppo/{k}"] = float(torch.mean(v, axis=0).detach().cpu().numpy()[0])
        timing["time/ppo/calc_stats"] = time.time() - t

        # ㉓ 额外指标
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        stats["ppo/kl_ref"] = float(kl_ref.item())  # 惩罚项（希望越小越好）
        stats["ppo/reward_all"] = float(reward_all.item())  # 真实奖励（希望越大越好）

        timing["time/ppo/total"] = time.time() - t0  # ㉔ 整个 step 总耗时
        stats.update(timing)  # ㉕ 合并到 stats 返回
        # NOTE: lr_scheduler.step() 在本函数外部调用
        return stats

    def batched_forward_pass(
            self,
            model: torch.nn.Module,  # 模型，用于生成输出
            queries: torch.Tensor,  # 输入的查询张量，形状 [B, T_q]
            responses: torch.Tensor,  # 输入的响应张量，形状 [B, T_r]
            model_inputs: dict,  # 其他模型输入数据，如 input_ids、attention_mask
            return_logits: bool = False,  # 是否返回 logits（原始预测）
            response_masks: Optional[torch.Tensor] = None,  # 响应部分的掩码，用于忽略填充部分
    ):
        """
        批量计算模型输出，处理过的 logprobs, logits, 以及 values 等。
        """
        bs = len(queries)  # batch size, 训练样本的数量
        fbs = self.config.mini_batch_size  # mini batch 大小
        all_logprobs = []  # 存储 log-probabilities (对数概率)
        all_logits = []  # 存储 logits（未归一化的概率）
        all_masks = []  # 存储有效 token 掩码
        all_values = []  # 存储值函数的输出

        # 批量迭代：将大的批次分成 mini-batches
        for i in range(math.ceil(bs / fbs)):  # 将 batch_size 拆分为 mini_batch_size
            input_kwargs = {key: value[i * fbs: (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs: (i + 1) * fbs]  # 获取当前 mini-batch 的查询数据
            response_batch = responses[i * fbs: (i + 1) * fbs]  # 获取当前 mini-batch 的响应数据
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs: (i + 1) * fbs]  # 获取响应掩码

            input_ids = input_kwargs["input_ids"]  # 获取 input_ids，形状 [B, T_q]
            attention_mask = input_kwargs["attention_mask"]  # 获取 attention_mask，形状 [B, T_q]

            # 前向传播，获取模型输出的 logits（logits是未经过 softmax 的原始输出）
            logits, _, values, _ = model(input_ids)  # shape: [B, T_r, vocab_size]

            # 计算 logprobs = log(π_θ) = log(π(s_t, a_t))
            # 这一步是 softmax 后的对数概率
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])  # shape: [B, T_r-1]
            masks = torch.zeros_like(attention_mask)  # 用于标记每个 token 是否有效
            masks[:, :-1] = attention_mask[:, 1:]  # 处理有效 token 掩码

            # 更新 masks，用于计算有效的 token 损失
            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1  # logprobs 从第二个 query token 开始
                if attention_mask[j, 0] == 0:  # 处理左边填充部分
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])  # 计算响应的结束位置
                if response_masks is not None:
                    response_masks_batch[j] = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]  # 处理响应的掩码

                masks[j, :start] = 0  # 忽略填充部分
                masks[j, end:] = 0  # 忽略后续无效部分
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:  # 是否返回 logits
                all_logits.append(logits)  # 存储 logits
            else:
                del logits  # 如果不需要 logits，删除
            all_values.append(values)  # 存储 values (V(s_t))
            all_logprobs.append(logprobs)  # 存储 logprobs
            all_masks.append(masks)  # 存储 masks

        # 拼接所有 mini-batch 的结果，返回
        return (
            torch.cat(all_logprobs),  # 拼接所有 logprobs，形状为 [B, T_r-1]
            torch.cat(all_logits)[:, :-1] if return_logits else None,  # 拼接 logits，如果需要的话
            torch.cat(all_values)[:, :-1],  # 拼接所有 value 输出，形状为 [B, T_r-1]
            torch.cat(all_masks)[:, :-1],  # 拼接所有 masks
        )


    '''
    score:奖励模型通常是预训练语言模型（如GPT、Llama）的微调版本，输入格式为(prompt, response)，输出一个标量分值。
    '''
    def compute_rewards(
            self,
            scores: torch.FloatTensor,  # 奖励模型的打分，形状 [B]
            logprobs: torch.FloatTensor,  # 当前模型的 logprobs，形状 [B, T-1]
            ref_logprobs: torch.FloatTensor,  # 参考模型的 logprobs，形状 [B, T-1]
            masks: torch.LongTensor,  # 有效 token 的掩码，形状 [B, T]
    ):
        """
        计算每个 token 的奖励，包含基于 KL 的惩罚和来自奖励模型的分数。
        """
        rewards, non_score_rewards = [], []  # 初始化奖励列表
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # 计算 KL 惩罚：KL = log(p_model / p_ref)
            kl = self._kl_penalty(logprob, ref_logprob)  # KL散度惩罚项
            non_score_reward = -self.kl_ctl.value * kl  # 基于 KL 惩罚的非得分奖励
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()  # 复制非得分奖励
            last_non_masked_index = mask.nonzero()[-1]  # 获取最后一个有效 token 索引

            # 将奖励模型分数加到最后有效 token 上
            reward[last_non_masked_index] += score
            rewards.append(reward)

        return torch.stack(rewards), torch.stack(non_score_rewards)  # 返回堆叠后的 rewards 和 non_score_rewards

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)
        raise NotImplementedError

    def compute_advantages(
            self,
            values: torch.FloatTensor,  # 估计的状态值 V(s_t)，形状 [B, T]
            rewards: torch.FloatTensor,  # 奖励，形状 [B, T]
            mask: torch.FloatTensor,  # 有效 token 掩码，形状 [B, T]
    ):
        """
        计算每个 token 的优势（Advantage），使用 GAE（广义优势估计）。
        """
        lastgaelam = 0
        advantages_reversed = []  # 用于存储反向计算的优势
        gen_len = rewards.shape[-1]  # 序列长度

        values = values * mask  # 将值函数值与掩码结合，忽略无效 token
        rewards = rewards * mask  # 将奖励与掩码结合，忽略无效 token

        '''
        1.为什么白化奖励和优势？
            稳定性：通过白化，我们去除了奖励和优势中的尺度差异，这样可以避免训练过程中某些较大的值主导模型学习过程，导致数值不稳定。
            加速收敛：数据具有零均值和单位方差时，优化器（如梯度下降）会更加高效，因为它可以避免某些维度上的梯度更新过大或过小。
        2.具体操作：
            奖励白化：首先计算奖励的均值和方差，然后将每个奖励值减去均值，除以标准差，得到零均值和单位方差的奖励数据。
            优势白化：与奖励白化类似，优势值也进行标准化，避免不同尺度的优势值对训练过程产生不平衡的影响。
        '''
        if self.config.whiten_rewards:  # 如果配置了白化奖励
            # 白化奖励：通过减去均值并除以标准差，将奖励标准化，使其具有零均值和单位方差
            rewards = masked_whiten(rewards, mask, shift_mean=False)  # shift_mean=False 表示不加回均值

        # 反向计算优势
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0  # 下一时间步的值
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            #A^t = ∑{l=0}^∞ (γλ)^l δ_{t+l}，其中δ_t = r_t + γV(s_{t+1}) − V(s_t)，γ是折扣因子，λ是平滑因子。
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam  # 计算广义优势估计（GAE）
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)  # 正序排列

        '''
        1. GAE 的数学原理
        GAE 的核心公式为递归形式：
        A^t_GAE = δ_t + (γλ) A^t+1_GAE
        其中:
            - δ_t = r_t + γV(s_t+1) - V(s_t): TD残差，衡量当前状态价值函数的预测误差
            - γ: 折扣因子 (0 < γ < 1)，控制未来奖励的当前价值
            - λ: 平滑因子 (0 ≤ λ ≤ 1)，平衡短期与长期优势估计的权重
        该公式的展开形式为：
        A^t_GAE = ∑_l=0^∞ (γλ)^l δ_t+l
        即对多步 TD 残差进行指数衰减加权，形成优势函数的估计。
        2. 参数的作用与选择
        (1) 折扣因子 γ
            - 作用：决定未来奖励的衰减速度
            - γ 越大，算法越关注长期回报；γ 越小，越侧重即时奖励
            - 典型值：通常设置为 0.99，以兼顾长期回报与短期收益
        (2) 平滑因子 λ
            - 作用：控制优势估计的"时间跨度":
                - λ=0: 仅使用单步 TD 残差（低方差高偏差）
                - λ=1: 退化为蒙特卡洛方法（高方差低偏差）
                - 0<λ<1: 在两者之间权衡，通常设置为 0.9~0.95
            - 优势：通过调整 λ，可以在复杂环境中优化策略更新的稳定性与收敛速度
        3. GAE 与回报计算
        GAE 的最终目标是估计每个时间步的回报（Return），其公式为：
        R_t = A^t_GAE + V(s_t)
        解释：回报 R_t 由两部分组成：
            - 优势函数 A^t_GAE: 衡量动作相对于状态价值的额外收益
            - 状态价值函数 V(s_t): 反映当前状态的长期预期价值
        意义：通过结合 GAE 的优势估计与值函数，策略梯度更新既能利用局部反馈（低方差），
        又能保留全局优化目标（低偏差）。
        '''
        returns = advantages + values  # 计算回报：R_t = A_t + V(s_t)
        advantages = masked_whiten(advantages, mask)  # 白化优势，使得优势也具有零均值和单位方差
        advantages = advantages.detach()  # detach，避免反向传播时修改优势

        return values, advantages, returns  # 返回值函数、优势和回报

    def compute_loss(
            self,
            old_logprobs: torch.FloatTensor,  # 旧策略 π_old 的 log-probabilities，形状 [B, T-1]
            values: torch.FloatTensor,  # 当前值函数预测 V(s_t)，形状 [B, T]
            logprobs: torch.FloatTensor,  # 当前策略 π_new 的 log-probabilities，形状 [B, T-1]
            logits: torch.FloatTensor,  # 当前策略 π_new 的 logits，形状 [B, T, V]
            vpreds: torch.FloatTensor,  # 预测的值函数 V(s_t)，形状 [B, T]
            mask: torch.LongTensor,  # 掩码，标记有效 token，形状 [B, T]
            advantages: torch.FloatTensor,  # 当前优势（Advantage），形状 [B, T]
            returns: torch.FloatTensor,  # 回报（Returns），形状 [B, T]
    ):
        """
        计算损失函数和统计量。
        """
        # ========== 计算值函数损失（vf_loss） ==========
        '''
        数学描述：
            值函数损失（vf_loss）是根据预测的状态值（V(s_t)）和实际回报（R_t）之间的差异来计算的：
            vf_loss = 0.5 * E[max((V(s_t) - R_t)², (V_clip(s_t) - R_t)²)]
            其中 V_clip(s_t) 是裁剪后的值函数预测。
            这个操作防止了值函数过度更新，使得损失不至于因为过大的差异而爆炸。
        '''
        # 将值函数的预测值裁剪到 V(s_t) ± cliprange_value 范围
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,  # clip 下界
            values + self.config.cliprange_value,  # clip 上界
        )

        # 计算值函数损失：对于每个 token，计算 V(s_t) 和回报 R_t 之间的误差
        vf_losses1 = (vpreds - returns) ** 2  # 预测值和回报的误差平方，损失 1
        vf_losses2 = (vpredclipped - returns) ** 2  # 裁剪后的预测值和回报的误差平方，损失 2
        # 取两者之间的最大值，防止过度更新
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)  # 加权计算损失
        # 计算裁剪比例，表示裁剪后损失占比
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)


        # ========== 计算策略损失（policy loss） ==========
        '''
        数学描述：
            策略损失（policy loss）是基于优势（Advantage）和比率（ratio）的：
            L_policy = E[min(ratio × A_t, clamp(ratio, 1-ε, 1+ε) × A_t)]
            ratio 是当前策略和旧策略的比率，A_t 是当前的优势（Advantage）。
            使用 clamp 进行比率裁剪，防止比率过大导致梯度爆炸。
        '''
        # 计算当前策略和旧策略的 log-probabilities 比率
        ratio = torch.exp(logprobs - old_logprobs)  # ratio = π_new(a_t|s_t) / π_old(a_t|s_t)



        # 计算政策损失：通过优势（Advantage）和比率（ratio）计算，策略损失是负的
        pg_losses = -advantages * ratio  # 原始政策损失
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange,
                                               1.0 + self.config.cliprange)  # 裁剪比率后的政策损失

        # 取两者的最大值，防止比率过大导致梯度爆炸
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)  # 策略损失
        # 计算裁剪比例，表示裁剪后损失占比
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        # 数学描述：
        # 策略损失（pg_loss）是根据优势（Advantage）和比率（ratio）计算的，具体如下：
        # pg_loss = E[min(ratio × A_t, clamp(ratio, 1-ε, 1+ε) × A_t)]
        # 其中，ratio 是当前策略和旧策略的比率。

        # ========== 计算总损失 ==========
        # 总损失 = 政策损失 + 值函数损失
        loss = pg_loss + self.config.vf_coef * vf_loss

        # 数学描述：
        # 总损失（loss）是策略损失（L_policy）和值函数损失（L_value）加权求和：
        # loss = L_policy + vf_coef × L_value
        # 其中 vf_coef 是值函数损失的权重系数。

        # ========== 检查比率是否过大（避免策略爆炸） ==========
        avg_ratio = masked_mean(ratio, mask).item()  # 计算平均比率
        # 如果比率超出阈值，则说明策略发生了剧烈变化，跳过当前 batch
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            # 如果比率过大，则将损失设置为零
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        # 数学描述：
        # 如果策略比率（ratio）超过某个阈值，表明策略的变化过大，会跳过当前批次的更新：
        # avg_ratio = 1/B * Σ ratio_i, 如果 avg_ratio 超过阈值，则跳过本次更新。

        # ========== 计算熵（entropy） ==========
        entropy = masked_mean(entropy_from_logits(logits), mask)  # 熵：模型的随机性，防止过度确定性

        # 数学描述：
        # 熵（entropy）用于衡量策略的不确定性，熵值较高表示策略较为随机（探索性强）：
        # entropy = - Σ p(a_i | s_t) log(p(a_i | s_t))
        # 熵值较高表示更多的探索，较低的熵值表示更多的利用。

        # ========== 计算 KL 散度（KL Divergence） ==========
        approxkl = 0.5 * masked_mean((old_logprobs - logprobs) ** 2, mask)  # 近似的 KL 散度，表示策略间的差异
        policykl = masked_mean(old_logprobs - logprobs, mask)  # 精确的 KL 散度

        # 数学描述：
        # KL 散度（KL Divergence）衡量旧策略和当前策略之间的差异：
        # KL = E[log(π_old(a_t | s_t)) - log(π_new(a_t | s_t))]
        # 这用来确保当前策略不会偏离旧策略太多，从而保持训练的稳定性。

        # ========== 计算回报（returns）和值函数（values）的统计量 ==========
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)  # 回报的均值和方差
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)  # 值函数的均值和方差

        # ========== 统计信息 ==========
        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )

        # 返回损失值和训练统计量
        loss_p, loss_v, train_stats = pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)
        return loss, train_stats

    def _early_stop(self, policykl):
        r"""
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL (策略的KL散度)

        Returns:
            `bool`: whether to early stop or not (是否提前停止训练)
        """
        early_stop = False  # 初始化提前停止标志

        # 如果没有启用提前停止，则直接返回 False
        if not self.config.early_stopping:
            return early_stop

        # 数学描述：策略 KL 散度（KL Divergence）的计算
        # KL散度用于衡量当前策略与参考策略之间的差异。如果策略的变化过大（即 KL 散度过大），则表明训练可能不稳定，此时我们会提前停止。
        # 假设当前的策略 KL 散度为 policy_kl，如果其大于设定的目标 KL 值（self.config.target_kl），则执行提前停止。
        # KL 散度计算如下：
        # KL = E[log(π_old(a_t | s_t)) - log(π_new(a_t | s_t))]  # 计算当前策略与旧策略之间的KL散度。

        # 如果没有使用分布式数据并行（DDP），直接根据策略KL是否大于目标KL来判断是否提前停止
        if not self.is_ddp and policykl > 1.5 * self.config.target_kl:  # 1.5 是一个超参数，表示允许的最大KL差异
            self.optimizer.zero_grad()  # 如果策略KL散度大于目标值，清空梯度
            early_stop = True  # 设置提前停止标志为 True

        # 如果使用分布式数据并行（DDP），则需要等待所有进程计算完毕，并进行跨进程的KL散度平均
        elif self.is_ddp:
            import torch.distributed as dist  # 导入分布式通信库

            # 等待所有进程计算完成（同步所有进程的状态）
            dist.barrier()

            # 跨进程汇总 KL 散度：通过 all_reduce 计算各个进程的 KL 散度之和
            dist.all_reduce(policykl, dist.ReduceOp.SUM)  # 使用求和操作，合并所有进程的KL散度
            policykl /= int(os.environ["WORLD_SIZE"])  # 将总和除以进程数量，得到平均KL值

            # 数学描述：在多GPU分布式训练时，所有进程的 KL 散度会被汇总并平均。然后，比较该平均 KL 散度与目标值的关系
            # 进行提前停止，防止训练过程中策略变化过大。
            if policykl > 1.5 * self.config.target_kl:  # 如果平均KL散度大于目标值的1.5倍，则提前停止训练
                self.optimizer.zero_grad()  # 清空梯度
                early_stop = True  # 设置提前停止标志为 True

        return early_stop  # 返回是否提前停止

def logprobs_from_logits(logits, labels, gather=True):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    计算 log-probabilities（对数概率）。
        logits 是模型输出的未经 softmax 处理的原始输出，labels 是目标标签。

    数学描述：
        log(π_θ(a_t | s_t)) = log(softmax(logits))，表示给定输入 logits，计算每个类别的对数概率。
        如果 gather=True，我们将从 logp 中选取与标签对应的对数概率。
    """
    logp = F.log_softmax(logits, dim=2)  # 计算 log(π_θ) = log(softmax(logits))，表示给定输入 logits 的对数概率

    if not gather:
        return logp  # 如果不需要从 logp 中获取特定标签的概率，直接返回 logp

    # 否则，从 logp 中选取与标签对应的对数概率
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)  # 按标签索引从 logits 中选取相应的对数概率
    return logpy


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    # 数学描述：
    # 对于带有掩码的值，掩码为 1 的部分会被计算在内，而掩码为 0 的部分会被忽略。
    # 对于给定的 mask，mean 计算如下：
    # E[\text{values}] = \frac{\sum_{i} \text{values}[i] \cdot \text{mask}[i]}{\sum \text{mask}[i]}

    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)  # 对指定轴计算均值
    else:
        return (values * mask).sum() / mask.sum()  # 对整个 tensor 计算均值


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    # 计算带有掩码的方差，忽略掩码为 0 的部分
    mean = masked_mean(values, mask)  # 先计算掩码后的均值
    centered_values = values - mean  # 将值减去均值，得到中心化的值
    variance = masked_mean(centered_values ** 2, mask)  # 计算方差：E[(x - \mu)^2]

    if unbiased:
        mask_sum = mask.sum()  # 获取掩码为 1 的样本数量
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`; "
                "try increasing the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # Bessel 修正：如果样本数量小于 N-1，则对方差进行无偏修正
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction  # 修正方差

    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)  # 计算均值和方差
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)  # 白化：去均值，除以标准差（避免除零）
    if not shift_mean:
        whitened += mean  # 如果不需要去除均值，可以加回来
    return whitened

def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extension to torch.clamp
    用于将输入张量限制在指定的最小值和最大值之间。

    数学描述：
    \text{clipped\_x} = \max(\min(x, \text{tensor\_max}), \text{tensor\_min})
    该操作将张量 x 限制在 tensor_min 和 tensor_max 的范围内，超出范围的值被裁剪为边界值。
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)  # 裁剪操作：将 x 限制在 [tensor_min, tensor_max] 之间
    return clipped



def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    # 计算 softmax 概率分布
    pd = torch.nn.functional.softmax(logits, dim=-1)  # 计算概率分布 p(a | s_t)

    # 数学描述：
    # 熵的计算公式：
    # H = - \sum_{i} p(a_i | s_t) \log(p(a_i | s_t))
    # 其中 p(a_i | s_t) 是每个动作 a_i 在状态 s_t 下的概率，log 是自然对数。

    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)  # 计算熵
    return entropy


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    # 数学描述：
    # 扁平化嵌套字典，将所有的嵌套键连接为单一的键。
    # 假设字典的嵌套层次为 N 层，那么每个嵌套键的连接将形成一个包含 N 个部分的键，
    # 其中每个部分用指定的分隔符（sep）连接。例如：
    # { "a": { "b": 1 }} 变成 "a/b": 1

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)  # 递归处理嵌套字典
            else:
                into[prefix + k] = v  # 将嵌套键连接成单一键

    flat = {}
    rec(nested, "", flat)  # 扁平化字典
    return flat


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        # 将每个字典的值展平并填充到一个 batch 中
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=-1)  # 将值堆叠在一起，并填充缺失值
    return results


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    该自适应 KL 控制器用于根据 KL 散度自适应地调整学习率。
    """

    def __init__(self, init_kl_coef, target, horizon):
        """
        初始化自适应 KL 控制器。

        参数：
        init_kl_coef (float): 初始 KL 系数。
        target (float): 目标 KL 散度值。
        horizon (int): 用于计算比例误差的时间范围。
        """
        self.value = init_kl_coef  # 初始的 KL 系数
        self.target = target  # 目标 KL 值
        self.horizon = horizon  # 控制比例变化的时间范围

    def update(self, current, n_steps):
        """
        根据当前 KL 值和时间步骤更新 KL 系数。

        参数：
        current (float): 当前的 KL 散度值。
        n_steps (int): 当前训练步骤数。
        """
        target = self.target
        # 计算比例误差，将当前 KL 值与目标 KL 值进行比较，限制误差范围在 [-0.2, 0.2] 之间
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)

        # 根据比例误差调整学习率
        mult = 1 + proportional_error * n_steps / self.horizon  # 调整比例
        self.value *= mult  # 更新 KL 系数

class FixedKLController:
    """
    数学描述：
    固定 KL 控制器的 KL 系数始终保持不变，更新操作是空操作。
    因此，KL 系数不受当前 KL 散度或训练步骤的影响。
    """

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class LengthSampler:
    """
    随机采样一个长度。
    """

    def __init__(self, min_value=0, max_value=10):
        """
        初始化长度采样器。

        参数：
        min_value (int): 采样的最小值。
        max_value (int): 采样的最大值。
        """
        self.values = list(range(min_value, max_value))  # 创建一个包含所有可能长度的列表

    def __call__(self):
        """
        随机选择一个长度。

        返回：
        int: 随机选出的长度。
        """
        return np.random.choice(self.values)  # 从可选长度中随机选择一个


class LengthReward:
    """
    根据序列长度计算奖励。
    """

    def __init__(self, target_length=200):
        """
        初始化长度奖励计算器。

        参数：
        target_length (int): 目标序列长度。
        """
        self.target_length = target_length  # 设置目标序列长度

    def __call__(self, sequence_length):
        """
        计算基于序列长度的奖励。

        参数：
        sequence_length (int): 当前序列的长度。

        返回：
        float: 计算得到的奖励值。
        """
        return -abs(self.target_length - sequence_length) / 100.  # 计算长度与目标长度的绝对差，并归一化