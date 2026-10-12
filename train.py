"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU/CPU, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py 
# 1. 预训练->ckpt.pt
python train.py \
    --out_dir="outmini" \
    --batch_size=128 \
    --max_seq_len=512 \
    --gradient_accumulation_steps=16 \
    --vocab_source="custom" \
    --vocab_size=512 \
    --dim=64 \
    --n_layers=5 \
    --n_heads=8 \
    --n_kv_heads=4 \
    --multiple_of=4 \
    --learning_rate=1e-3 \
    --dropout=0.05 \
    --weight_decay=0.01 \
    --max_iters=100000 \
    --beta2=0.99 \
    --warmup_iters=1000 \
    --eval_interval=2000 \
    --eval_iters=100 \
    --compile
    
# 2.基于预训练模型ckpt.pt，PPO微调
python train.py     \
    --out_dir="stories260K"     \
    --batch_size=50     \
    --max_seq_len=512     \
    --gradient_accumulation_steps=1     \
    --vocab_source="custom"     \
    --vocab_size=512     \
    --dim=64     \
    --n_layers=5     \
    --n_heads=8     \
    --n_kv_heads=4     \
    --multiple_of=4     \
    --learning_rate=1e-4     \
    --dropout=0.00     \
    --weight_decay=0.01     \
    --max_iters=98049     \
    --beta2=0.99     \
    --warmup_iters=1000     \
    --eval_interval=20     \
    --eval_iters=5     \
    --compile=True    \
    --device=cuda    \
    --eval_only=False   \
    --init_from="resume" \
    --ppo=True  \
    --decay_lr=False  \
    --always_save_checkpoint=True  \
    --start_len=30

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""


"""
这段代码是用来训练TinyStories数据集的模型，包含了预训练和基于PPO的微调两个主要阶段。它支持单机单卡训练以及分布式训练。通过多种技术手段，如学习率调度、梯度累积、模型编译等来优化训练过程。

该代码的训练过程包括以下几个主要步骤：

1. **预训练阶段（预设参数）**：
   - 该阶段的目标是从头开始训练一个Transformer模型，直到达到指定的迭代次数。
   - 预训练配置包括批次大小、学习率、训练轮数、模型的维度和层数等。

2. **PPO微调阶段（使用预训练模型）**：
   - 在预训练阶段完成后，使用预训练好的模型来进行强化学习的微调。
   - 该阶段采用PPO（Proximal Policy Optimization）策略，通过奖励机制不断调整模型参数，提高模型在给定任务上的表现。

3. **分布式训练（DDP）**：
   - 当使用多个GPU进行训练时，通过`DistributedDataParallel (DDP)`来实现多GPU并行训练，确保不同GPU上的模型参数同步更新。

4. **优化技术**：
   - 使用学习率预热（warmup）和余弦衰减调度来平滑调整学习率。
   - 使用梯度累积（gradient accumulation）来应对大批量训练时显存不足的问题。
   - 使用混合精度训练（AMP）来加速训练，并减少显存消耗。

5. **检查点与评估**：
   - 在训练过程中，定期保存模型检查点，并在验证集上进行评估，确保模型的性能。
"""

# -----------------------------------------------------------------------------
# 👉 依赖与通用工具
# -----------------------------------------------------------------------------
import math                  # 引入“数学工具箱”，提供对数/三角等函数
import os                    # 跟操作系统打交道（建文件夹、读环境变量…）
import time                  # 计时用：time.time() 得到当前秒
import json                  # 把 Python 对象 ↔ JSON 字符串
from copy import deepcopy    # 深复制：PPO 里要复制整张模型
from contextlib import nullcontext  # “什么也不做”的 with 占位符
from datetime import datetime       # 获取当前日期时间
from functools import partial       # 提前锁定部分参数，得到“半成品函数”
from dataclasses import dataclass, asdict, is_dataclass
from typing import List           # 🛠️ 类型标注：奖励函数接收 token 列表

import numpy as np           # 🛠️ 新增：PPO 日志里统计奖励均值/标准差
import torch                 # PyTorch 主库，负责张量+自动求导
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# ----- 项目本地依赖 -----
from model import (          # 本项目自带的模型文件
    Transformer,             #   标准 Decoder-only Transformer
    TransformerWithValueHead,#   多了一个 value head，PPO 用来估值
    ModelArgs,               #   用来打包模型超参
)
from tinystories import Task, get_tokenizer_model_path  # TinyStories 数据迭代器与词表文件定位
from tokenizer import Tokenizer                         # 把 SentencePiece 打包成简单接口
from export import model_export                         # 训练完把权重导出成 model.bin
# 🛠️ 新增：PPO 训练器与奖励函数（原来 train.py 只 import 了模型类，ppo.py 从未被调用）
from ppo import PPOTrainer, PPOConfig, LengthReward, LengthSampler
import argparse
# -----------------------------------------------------------------------------
# 👉 1. 可调“旋钮”汇总（保持原注释，仅分组整理）
# -----------------------------------------------------------------------------
# @dataclass
def TrainConfig():
    # 创建带分组显示的解析器
    parser = argparse.ArgumentParser(
        description="TinyStories 训练参数配置",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 自动显示默认值
    )

    # ---------------------------
    # 参数分组：日志与输出
    # ---------------------------
    log_group = parser.add_argument_group("日志与输出")
    log_group.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    log_group.add_argument("--out_dir", type=str, default="out", help="输出目录路径")  # 🛠️ 补默认值
    log_group.add_argument("--eval_only", action="store_true", default=False, help="仅执行评估模式")
    log_group.add_argument("--eval_iters", type=int, default=100, help="验证迭代次数") #每次迭代取batch数据
    log_group.add_argument("--eval_interval", type=int, default=2000, help="验证间隔步数") 
    log_group.add_argument("--always_save_checkpoint", action="store_true", default=False, help="始终保存检查点")
    log_group.add_argument("--init_from", type=str, choices=["scratch", "resume"], default="scratch", help="初始化:从头训/接续") #resume
    # 🛠️ 新增：显式指定续训 checkpoint（跨目录续训、加载别人给的 ckpt 时很有用）
    log_group.add_argument("--ckpt_path", type=str, default="", help="续训 checkpoint 路径（留空则自动在 out_dir 中查找）")
    # 🛠️ 新增：把 --max_iters 解释为「本轮再训多少步」而不是「总步数」。
    #    resume 时 ckpt 里的 iter_num 往往已经很大（如 98000），若 max_iters 仍按“总步数”
    #    解释，会出现两种静默故障：① 一步都不跑就退出；② get_lr() 落到 min_lr=0，
    #    学习率变成 0、模型根本不更新（README 的 PPO 示例参数正好会踩到）。
    log_group.add_argument("--iters_from_now", action="store_true", default=False,
                          help="--max_iters 表示“从当前步开始再训多少步”（默认表示总步数）")

    # ---------------------------
    # 参数分组：数据配置
    # ---------------------------
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    data_group.add_argument("--max_seq_len", type=int, default=256, help="序列最大长度")
    data_group.add_argument("--vocab_source", type=str, choices=["llama2", "custom"], 
                          default="llama2", help="词表来源")
    data_group.add_argument("--vocab_size", type=int, default=32000, help="词表尺寸（需与来源一致）")
    data_group.add_argument("--tokenizer_path", type=str, default="", help="自定义分词器路径")
    # 🛠️ 新增：数据加载并行度。原来硬编码 num_workers=0，读盘/反序列化与训练抢同一线程，
    #    GPU 空转。CPU 核心多的话设 2~8（注意：worker 多会多吃内存，因为每个 worker 一份 bin 映射）。
    data_group.add_argument("--num_workers", type=int, default=0, help="DataLoader 工作进程数（0=主线程读取）")

    # ---------------------------
    # 参数分组：模型结构
    # ---------------------------
    model_group = parser.add_argument_group("模型架构")
    model_group.add_argument("--dim", type=int, default=288, help="隐藏层维度")
    model_group.add_argument("--n_layers", type=int, default=6, help="Transformer层数")
    model_group.add_argument("--n_heads", type=int, default=6, help="注意力头数")
    model_group.add_argument("--n_kv_heads", type=int, default=6, help="KV头数（分组查询）")
    model_group.add_argument("--multiple_of", type=int, default=32, help="FFN维度倍数")
    model_group.add_argument("--dropout", type=float, default=0.0, help="Dropout概率")

    # ---------------------------
    # 参数分组：优化器
    # ---------------------------
    optim_group = parser.add_argument_group("优化器参数")
    optim_group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    optim_group.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    optim_group.add_argument("--max_iters", type=int, default=100000, help="最大迭代次数")
    optim_group.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减系数")
    optim_group.add_argument("--beta1", type=float, default=0.9, help="Adam beta1参数")
    optim_group.add_argument("--beta2", type=float, default=0.95, help="Adam beta2参数")
    optim_group.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # ---------------------------
    # 参数分组：学习率调度（修正布尔逻辑）
    # ---------------------------
    lr_group = parser.add_argument_group("学习率调度")
    # 🛠️ 默认值改成 None，用来区分「用户显式关闭衰减」和「没传」：
    #    * 预训练：不传 → 启用余弦衰减（与原默认一致）
    #    * PPO  ：不传 → 不衰减（RL 微调通常用固定小学习率；否则 resume 时
    #             余弦调度会算出接近 min_lr 甚至 0 的学习率，--ppo_lr 形同虚设）
    lr_group.add_argument("--decay_lr", action="store_false", default=None,
                    help="禁用学习率衰减（预训练默认启用，PPO 默认禁用）")  
    '''学习率预热"在训练的最开始阶段，逐步增加学习率到一个预设的最大值，而不是一开始就使用较大的学习率。
    这样做的主要目的是稳定训练过程，避免模型参数在初始阶段因为过大的学习率而发生剧烈波动，导致模型不收敛或收敛缓慢。'''
    lr_group.add_argument("--warmup_iters", type=int, default=1000, help="学习率预热步数")
    lr_group.add_argument("--min_lr", type=float, default=0.0, help="最低学习率")

    # ---------------------------
    # 参数分组：设备配置（修正布尔型参数）
    # ---------------------------
    device_group = parser.add_argument_group("硬件配置")
    device_group.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], 
                            default="cuda", help="计算设备")
    # 🛠️ 原来 choices 只有 float16/bfloat16，但 _setup_ddp 里的精度映射表是有 float32 的：
    #    CPU 训练/调试想指定 float32 会被 argparse 直接拒绝（invalid choice）。
    device_group.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], 
                            default="float16", help="数据类型")
    device_group.add_argument("--compile", action="store_true",default=False,  # 🛠️ 改为布尔标记
                            help="启用模型编译优化（默认禁用）") 

    # ---------------------------
    # 参数分组：PPO 配置（修正默认逻辑）
    # ---------------------------
    ppo_group = parser.add_argument_group("强化学习（PPO）")
    ppo_group.add_argument("--ppo", action="store_true", default=False, help="启用PPO训练")
    ppo_group.add_argument("--no_ppo_debug", action="store_false", dest="ppo_debug",  # ✅ 反向标记
                          default=True, help="禁用PPO调试模式（默认启用）")  
    ppo_group.add_argument("--init_kl_coef", type=float, default=0.2, help="初始KL系数")
    ppo_group.add_argument("--target_kl", type=float, default=0.2, help="目标KL散度")
    ppo_group.add_argument("--start_len", type=int, default=30, help="初始生成长度")
    ppo_group.add_argument("--target_len", type=int, default=200, help="目标序列长度")
    ppo_group.add_argument("--max_gen_len", type=int, default=300, help="最大生成长度")

    # 🛠️ 新增：把 ppo.py 的 PPOTrainer 真正接进训练循环所需的参数
    ppo_group.add_argument("--ppo_rollout_size", type=int, default=64,
                          help="每个 PPO step 采样的 (prompt, response) 对数（不能超过 --batch_size）")
    ppo_group.add_argument("--ppo_epochs", type=int, default=4, help="同一批 rollout 复用几个 epoch")
    ppo_group.add_argument("--ppo_mini_batch_size", type=int, default=0,
                          help="PPO 的 mini-batch 大小（0 = 自动 = rollout_size/8）")
    ppo_group.add_argument("--ppo_grad_accum_steps", type=int, default=1, help="PPO 梯度累积步数")
    ppo_group.add_argument("--ppo_lr", type=float, default=0.0,
                          help="PPO 阶段学习率（0 = 沿用 --learning_rate）")
    ppo_group.add_argument("--ppo_temperature", type=float, default=0.9, help="rollout 采样温度")
    ppo_group.add_argument("--ppo_top_k", type=int, default=0, help="rollout top-k（0=关闭）")
    ppo_group.add_argument("--ppo_top_p", type=float, default=0.95, help="rollout top-p（<=0=关闭）")
    ppo_group.add_argument("--ppo_cliprange", type=float, default=0.2, help="策略比率裁剪范围 ε")
    ppo_group.add_argument("--ppo_cliprange_value", type=float, default=0.2, help="value 裁剪范围")
    ppo_group.add_argument("--ppo_vf_coef", type=float, default=0.1, help="value loss 权重")
    ppo_group.add_argument("--ppo_kl_penalty", type=str, default="kl",
                          choices=["kl", "abs", "mse", "full"], help="KL 惩罚形式")
    ppo_group.add_argument("--ppo_gamma", type=float, default=1.0, help="GAE 折扣因子 γ")
    ppo_group.add_argument("--ppo_lam", type=float, default=0.95, help="GAE 平滑因子 λ")
    ppo_group.add_argument("--ppo_early_stopping", action="store_true", default=False,
                          help="KL 超过 1.5×target_kl 时提前结束该 batch 的 epoch")
    ppo_group.add_argument("--ppo_whiten_rewards", action="store_true", default=False,
                          help="对奖励做白化（零均值单位方差）")
    ppo_group.add_argument("--ppo_reward", type=str, default="length",
                          help="奖励函数，可逗号组合：length（长度对齐）/ completion（结尾标点）/ "
                               "repetition（n-gram 不重复率）；也支持 both(=length+completion)、all（三者相加）")
    ppo_group.add_argument("--ppo_reward_target_len", type=int, default=0,
                          help="长度奖励的目标长度（0 = 用 --target_len）")
    ppo_group.add_argument("--ppo_gen_len", type=int, default=0,
                          help="rollout 生成长度（0 = 自动 = min(max_gen_len, max_seq_len-start_len-1)）")
    # 🛠️ PPO 质量监控的评估批量：训练用的 batch_size 默认 128、max_seq_len 默认 256，
    #    一次评估前向在 2 核 CPU 上要 ~3 秒；监控只需要看趋势，16×256 就够（快一个量级）。
    ppo_group.add_argument("--ppo_eval_batch_size", type=int, default=16,
                          help="PPO 质量监控（CE 损失）用的小批量")
    ppo_group.add_argument("--ppo_prompt_file", type=str, default="",
                          help="prompt 文本文件（每行一条故事开头）；留空则从训练集取真实故事开头。"
                               "每行会被编码并截断到 --start_len 个 token")

    arg = parser.parse_args()
    # 🛠️ 归一化 --decay_lr：None 表示“未指定”，按阶段取默认值（预训练 True / PPO False）
    if arg.decay_lr is None:
        arg.decay_lr = not arg.ppo
    return arg


# -----------------------------------------------------------------------------
# 👉 2. 工具函数：学习率调度 & 评估
# -----------------------------------------------------------------------------
import math

def get_lr(it: int, arg: TrainConfig, max_iters: int = None) -> float:
    """线性 warmup + 余弦衰减调度器
    
    参数说明：
    it - 当前迭代步数（第几次参数更新）
    arg - 训练配置对象，包含以下属性：
        warmup_iters: 预热阶段的迭代次数
        max_iters:    最大迭代次数
        learning_rate:基础学习率
        min_lr:       最小学习率（衰减下限）
    """
    
    # ==================== 预热阶段 ====================
    # 当迭代步数小于预热步数时，执行线性增长策略
    # 公式：lr = base_lr * (当前步数 / 总预热步数)
    # 作用：避免训练初期参数剧烈波动，类似「缓步加速」
    # 🛠️ 支持显式传入衰减终点：resume / PPO 阶段需要用它，否则 it > max_iters 时
    #    会直接返回 min_lr（默认 0）→ 学习率静默变 0，训练“看似在跑但没有更新”。
    max_iters = arg.max_iters if max_iters is None else max_iters
    if it < arg.warmup_iters:
        return arg.learning_rate * it / arg.warmup_iters
    
    # ================ 训练终止：保护机制 ================
    # 当超过最大迭代步数时保持最小学习率
    # 作用：防止后续计算产生意外波动
    if it > max_iters:
        return arg.min_lr
    
    # ============== 训练中：余弦衰减阶段（核心计算） ==============
    # 计算衰减进度：[0,1]区间，0表示刚结束预热，1表示训练结束
    # 示例：当 warmup=1000, max=10000, it=2000 时：
    # decay_ratio = (2000-1000)/(10000-1000) ≈ 0.111
    decay_ratio = (it - arg.warmup_iters) / max(1, max_iters - arg.warmup_iters)
    
    # 生成余弦系数：通过余弦函数实现平滑过渡
    # 曲线特性：从1（起始点）逐渐下降至0（结束点）
    # 公式推导：将标准余弦函数cos(πx)的值域[-1,1]映射到[0,1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 当decay_ratio=0→1, coeff=1→0
    
    # 最终学习率计算：线性插值公式
    # 基础公式：最终值 = 最小值 + 衰减系数 * (最大值 - 最小值)
    # 动态范围：[min_lr, learning_rate] 随coeff从1→0而衰减
    return arg.min_lr + coeff * (arg.learning_rate - arg.min_lr)



                           
'''
评估流程：
[数据加载器] → [批次数据X/Y] → [模型预测] → [计算loss] → [重复N次取平均]
           ↖_________________________↙
'''
@torch.no_grad()  # 魔法开关：关闭自动求导功能，就像考试时不需要记笔记
def estimate_loss(model, next_batch, ctx, arg: TrainConfig):
    """
    评估模型在训练集和验证集上的平均损失（相当于模拟考试）

    🛠️ 参数由 iter_batches(生成器工厂) 改为 next_batch(split)->(X,Y)，
       以便复用「长生命周期的数据迭代器」，避免每次评估都重建 DataLoader。
    """
    # 结果字典：存放训练集和验证集的平均损失
    out = {}
    
    # 切换模型为考试模式：关闭 dropout / LayerNorm 统计等训练专用功能，类似关闭手机专心考试
    model.eval()  
    
    # 同时在训练集和验证集上测试，好比同时做课堂练习和期末考试
    for split in ["train", "val"]:  
        # 创建存放多次考试结果的成绩单（arg.eval_iters次测试取平均）
        losses = torch.zeros(arg.eval_iters)  
        
        # 进行多次小测试取平均（避免单次考试的偶然性）
        for k in range(arg.eval_iters):
            # 获取一个批次的题目和答案（X是题目，Y是标准答案）
            X, Y = next_batch(split)  
            # 使用混合精度计算（相当于用更高效的答题工具）
            with ctx:  # fp16 / bf16 
                # 模型做题：获取预测结果和错误程度（logits是预测答案，loss是错误值）
                _, loss = model(X, Y)[-2:]  
            # 记录本次测试的错误值（把成绩记到成绩单里）
            losses[k] = loss.item()  
        
        # 计算平均错误值（全班多次考试的平均分）
        out[split] = losses.mean()  
    
    # 恢复模型为学习模式：重新开启训练专用功能，就像下课继续学习
    model.train()  
    
    return out  # 返回两份成绩单：训练集和验证集的平均错误值


# -----------------------------------------------------------------------------
# 👉 2.5 工具函数：配置序列化 / 检查点定位 / 权重前缀清理
# -----------------------------------------------------------------------------
def config_to_dict(arg) -> dict:
    """
    把配置对象安全地转成 dict，用于写进 checkpoint。

    🛠️ 原代码用的是 dataclasses.asdict(arg)，但 TrainConfig 只是一个
    argparse.Namespace（返回 parser.parse_args()），并不是 dataclass：
        >>> asdict(Namespace(a=1))
        TypeError: asdict() should be called on dataclass instances
    也就是说：第一次触发“保存 checkpoint”时程序必然崩溃，训练白跑。
    这里统一用 vars()（Namespace）/ dataclasses.asdict()（dataclass）两种分支。
    """
    if is_dataclass(arg) and not isinstance(arg, type):
        return asdict(arg)
    if isinstance(arg, argparse.Namespace):
        return vars(arg).copy()
    return dict(vars(arg))


def resolve_ckpt_path(out_dir: str, ppo: bool, ckpt_path: str = "") -> str:
    """
    定位续训用的 checkpoint 文件。

    🛠️ 原代码在 init_from=resume 时只会去找 out_dir/ckpt.pt，但：
      * PPO 阶段保存的文件名是 ppo_ckpt.pt（见 train() 里的 fname 逻辑）；
      * README 里 PPO 阶段又把 out_dir 指到了另一个目录（outmini vs stories260K）。
    结果就是“续训/PPO 微调”阶段直接 FileNotFoundError。
    现在按 ckpt_path > ppo_ckpt.pt > ckpt.pt 的顺序查找，仍然找不到再报错。
    """
    if ckpt_path:
        assert os.path.isfile(ckpt_path), f"指定的 checkpoint 不存在：{ckpt_path}"
        return ckpt_path
    candidates = ["ppo_ckpt.pt", "ckpt.pt"] if ppo else ["ckpt.pt", "ppo_ckpt.pt"]
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            return path
    return os.path.join(out_dir, candidates[0])


def model_args_to_dict(model) -> dict:
    """取模型真实的结构超参（ModelArgs 字段），用于写进 checkpoint。"""
    params = getattr(model, "params", None)
    if params is None:                      # DDP / compile 包装时剥一层
        params = getattr(getattr(model, "module", None), "params", None)
    if params is None:
        return {}
    return {k: v for k, v in dict(vars(params)).items() if not k.startswith("_")}


def checkpoint_args(arg, model) -> dict:
    """
    组装写进 checkpoint 的 model_args：
    训练配置（arg）+ 模型真实结构，后者优先级更高，保证权重形状可被正确重建。
    """
    merged = config_to_dict(arg)
    merged.update(model_args_to_dict(model))
    return merged


def load_model_state(model, state_dict: dict, master_process: bool = True):
    """
    加载权重，并处理 torch.compile 留下的 `_orig_mod.` 前缀。

    🛠️ 这是最隐蔽的一个坑：README 里预训练用了 --compile，
    torch.compile 会把参数名改成 `_orig_mod.xxx` 存进 ckpt，
    而 resume 时模型没有 compile → strict=False 下所有 key 都“对不上”，
    一句 warning 都不报就把权重全丢了：所谓的 PPO 微调其实是**随机初始化**在训练。
    这里既兼容带前缀的 ckpt，也兼容不带前缀的 ckpt，并显式校验加载比例。
    """
    prefix = "_orig_mod."
    has_prefixed = any(k.startswith(prefix) for k in state_dict)
    has_plain = any(not k.startswith(prefix) for k in state_dict)
    if has_prefixed and not has_plain:
        state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    info = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(info.missing_keys), list(info.unexpected_keys)
    if master_process:
        print(f"✓ 权重装载完成：缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
        if missing:
            print(f"  ⚠️ 缺失（未从 ckpt 恢复）：{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  ⚠️ 多余（ckpt 里有但模型没有）：{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    # 只允许 freqs_* 这类非参数 buffer 缺失，否则说明权重根本没加载上
    real_missing = [k for k in missing if "freqs" not in k and "cache" not in k]
    if len(real_missing) > 0.5 * len(list(model.state_dict())):
        raise RuntimeError(
            f"⚠️ 超过一半的参数没有从 checkpoint 恢复（缺失 {len(real_missing)} 项），"
            f"请检查 checkpoint 与模型结构是否一致，避免静默地从头训练。"
        )
    return info


def build_reward_fn(kind: str, target_len: int, enc, sampler=None):
    """
    构造「奖励函数」：`reward(tokens) -> float`。

    🛠️ 原仓库的 reward 只有 ppo.py 里的 LengthReward，但没有任何地方调用它，
       也就没有奖励信号。这里把它接上，并额外提供更适合 TinyStories 的信号：
         * length     ：`LengthReward`（-|目标长度 - 实际长度| / 100），控制篇幅
         * completion ：结尾是否落在句末标点（. ! ? " '），鼓励生成完整故事而不是半句话
         * both       ：两者相加
    `sampler` 非空时，每个样本都会重新采样一次目标长度（用 ppo.py 自带的 LengthSampler），
    让模型学会生成不同篇幅的故事。

    需要自定义奖励（例如接 reward model / 人工偏好数据）时，保持 `__call__(tokens) -> float`
    这个签名，替换传进来的 reward_fn 即可。
    """
    return TinyStoriesReward(kind=kind, target_len=target_len, enc=enc, sampler=sampler)


class TinyStoriesReward:
    """
    基于「长度 + 结尾完整性 + 重复度」的奖励函数（可任意组合）。

    为什么要有 repetition 项：TinyStories 这种小模型最典型的质量问题就是**复读**
    （“She was happy. She was happy. She was happy.”）。这项奖励是密集的、
    随时可优化的，比“结尾标点”这种稀疏信号更适合小规模 PPO 实验。
    """

    #: 支持的分项与别名
    ALIASES = {"both": ("length", "completion"), "all": ("length", "completion", "repetition")}

    def __init__(self, kind: str = "length", target_len: int = 200, enc=None, sampler=None,
                 ngram: int = 3):
        # sampler 定义自 ppo.py 自带的工具，采样一次得到本次的目标长度
        self.kind = kind
        self.target_len = target_len
        self.enc = enc
        self.sampler = sampler
        self.ngram = ngram
        self.kinds = self.parse_kinds(kind)
        self.length_reward = LengthReward(target_length=target_len)

    @classmethod
    def parse_kinds(cls, kind: str):
        """把 'repetition,length' / 'both' / 'all' 这类写法解析成分项列表。"""
        parts = []
        for raw in str(kind).split(","):
            raw = raw.strip().lower()
            if not raw:
                continue
            parts.extend(cls.ALIASES.get(raw, (raw,)))
        valid = {"length", "completion", "repetition"}
        unknown = [p for p in parts if p not in valid]
        assert not unknown, f"未知的奖励项 {unknown}，可选：{sorted(valid | set(cls.ALIASES))}"
        return tuple(dict.fromkeys(parts))  # 去重且保序

    def repetition_score(self, tokens: List[int]) -> float:
        """n-gram 唯一率：1.0 = 完全没有重复，越低说明复读越严重。"""
        n = max(1, self.ngram)
        if len(tokens) < n:
            return 1.0
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        return len(set(grams)) / len(grams)
    
    def current_target(self) -> int:
        """当前样本的目标长度：有 sampler 就采样，否则用固定值。"""
        return int(self.sampler()) if self.sampler is not None else self.target_len

    def length_score(self, tokens: List[int]) -> float:
        self.length_reward.target_length = self.current_target()
        return float(self.length_reward(len(tokens)))

    def completion_score(self, tokens: List[int]) -> float:
        text = self.enc.decode(tokens).strip()
        return 1.0 if text.endswith((".", "!", "?", '"', "'")) else 0.0

    def __call__(self, tokens: List[int]) -> float:
        score = 0.0
        for kind in self.kinds:
            if kind == "length":
                score += self.length_score(tokens)
            elif kind == "completion":
                score += self.completion_score(tokens)
            elif kind == "repetition":
                score += self.repetition_score(tokens)
        return score


# -----------------------------------------------------------------------------
# 👉 3. Trainer 封装
# -----------------------------------------------------------------------------
class TinyStoriesTrainer:
    """把原脚本逻辑封装进一个类，便于后续扩展（多机、超参搜索等）。"""


    def __init__(self, arg: TrainConfig):
        self.arg = arg             # 参数类封装
        self._setup_ddp()          # DDP / 随机种子 / TF32
        self._build_dataloader()   # 训练/验证数据迭代器
        self._build_model()        # GPT 或 GPT+V-head
        self._build_optimizer()    # AdamW / GradScaler
        if arg.ppo:                # 🛠️ 启用 PPO 时再组建 PPO 训练器（含冻结的参考模型）
            self._build_ppo()
        # 🛠️ 统一算出「本轮的目标步数」：默认 max_iters 是总步数；
        #    加了 --iters_from_now 则从当前 iter_num 起算（PPO 阶段还会自动兜底，见 train_ppo）。
        self.start_iter = getattr(self, "iter_num", 0)
        self.target_iters = self.start_iter + arg.max_iters if arg.iters_from_now else arg.max_iters

    # --------------------- 分布式环境相关 ---------------------
    # --------------------- 分布式训练环境设置（类似厨房团队分工） ---------------------
    def _setup_ddp(self):
        arg = self.arg
        
        # 判断是否启用分布式训练（检查是否有RANK环境变量）
        # 类比：检查是否有团队分工表（RANK存在表示需要分工协作）
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        
        if self.ddp:                                         #需要使用torchrun正确启动
            # 初始化进程组（组建厨房团队），使用NCCL通信后端
            # 类比：给所有厨师配发对讲机（nccl是高效通信协议）
            init_process_group(backend="nccl")
            
            # 获取当前厨师的编号和总人数
            self.ddp_rank = int(os.environ["RANK"])        # 全局编号（0是主厨）
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"]) # 本地厨房编号
            self.ddp_world_size = int(os.environ["WORLD_SIZE"]) # 总厨师数
            print("self.ddp_world_size",self.ddp_world_size)
            # 🛠️ 原代码这里有一句 exit()：只要用 torchrun 多卡启动就会立刻退出进程，
            #    DDP 等于完全不可用（且 exit() 只存在于交互解释器，脚本里应使用 sys.exit）。
            #    已删除，让下面的设备绑定 / DDP 包装正常执行。
            
            # 指定当前厨师使用的灶台（GPU设备）
            arg.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(arg.device)  # 绑定灶台
            
            # 判断是否是主厨（只有主厨负责记录菜谱和汇报）
            self.master_process = self.ddp_rank == 0
            
            # 设置随机种子偏移量（不同厨师用不同随机顺序处理食材）
            self.seed_offset = self.ddp_rank
            
            # 确保梯度累积步数能被厨师数整除（均匀分配工作量）
            # 例如：总步数100，4个厨师 → 每人25步
            assert arg.gradient_accumulation_steps % self.ddp_world_size == 0
            arg.gradient_accumulation_steps //= self.ddp_world_size
        else:
            # 单卡训练模式（整个厨房只有一个厨师）
            self.master_process = True    # 自己就是主厨
            self.seed_offset = 0          # 不需要随机偏移
            self.ddp_world_size = 1       # 团队只有1人

        # --------------------- 随机种子设置（保证实验可复现） ---------------------
        # 主厨负责创建菜谱保存目录
        if self.master_process:
            os.makedirs(arg.out_dir, exist_ok=True)  # 创建保存模型的文件夹
            
        # 设置随机种子（让不同厨师的切菜顺序不同但可控）
        # 基础种子1337 + 厨师编号 → 保证不同进程数据不同
        torch.manual_seed(1337 + self.seed_offset)      # CPU随机数
        torch.cuda.manual_seed(1337 + self.seed_offset) # GPU随机数
        
        # 启用TF32数学计算模式（加速计算但精度略有降低）
        # 类比：使用快速切菜刀（效率高但切口稍粗糙）
        torch.backends.cuda.matmul.allow_tf32 = True  # 矩阵乘法
        torch.backends.cudnn.allow_tf32  = True       # 卷积运算
        # 🛠️ 新增：让 fp32 矩阵乘法也走 TF32（新式 API，效果同 allow_tf32，但覆盖更全；
        #    对 CPU 无副作用），这与上面的两行一起才能完全打开 TF32 加速。
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        
        # 记录设备类型（cuda或cpu）
        self.device_type = "cuda" if "cuda" in arg.device else "cpu"

        # --------------------- 混合精度训练配置（平衡速度与精度） ---------------------
        # 根据配置选择数据类型（像选择不同精度的称量工具）
        ptdtype = {
            "float32": torch.float32,   # 高精度模式（电子秤）
            "bfloat16": torch.bfloat16, # 平衡模式（普通厨房秤）
            "float16": torch.float16    # 快速模式（粗略估算）
        }[arg.dtype]
        
        # 创建自动混合精度上下文（根据设备类型自动切换）
        # nullcontext表示无操作（在CPU上不需要特殊处理）
        self.ctx = (
            nullcontext() if self.device_type == "cpu"
            else torch.amp.autocast(
                device_type=self.device_type,
                dtype=ptdtype
            )
        )  # 进入这个上下文后，PyTorch会自动选择合适精度计算


    # --------------------- 数据 ---------------------
    # --------------------- 数据加载器构建（像快递分拣流水线） ---------------------
    def _build_dataloader(self):
        # 🛠️ 原方法名是 _build_dloader，但 __init__ 里调用的是 self._build_dataloader()
        #    → 一启动就 AttributeError: 'TinyStoriesTrainer' object has no attribute
        #    '_build_dataloader'，train.py 根本无法运行（除 --help 外任何参数都会崩）。
        #    这里改回正确的名字，并保留 _build_dloader 作为向后兼容别名。
        # 获取配置参数（相当于快递公司的操作手册）
        arg = self.arg  # 从类实例中取出预设参数

        # 创建批数据生成器（设置快递分拣规则）
        self.iter_batches = partial(  # partial像预填表格，固定部分参数
            Task.iter_batches,        # 核心数据生成器（主分拣机）

            # 以下是分拣规则参数：
            batch_size=arg.batch_size,      # 每箱快递数量（如32件/箱）
            max_seq_len=arg.max_seq_len,    # 单件最大尺寸（如256字符）
            vocab_size=arg.vocab_size,      # 快递类型总数（如5000种）
            vocab_source=arg.vocab_source,  # 快递来源（如"自定义"或"公开数据集"）
            device=arg.device,              # 运送工具（如cuda:0是1号卡车）
            # 🛠️ num_workers 原来硬编码为 0：读盘 + 反序列化完全跑在主线程上，
            #    GPU 会周期性空转等数据。改为可通过 --num_workers 配置。
            num_workers=arg.num_workers,
        )
        # 🛠️ 关键性能修复：Task.iter_batches() 每调用一次都会新建一个 DataLoader，
        #    num_workers>0 时还会重新 fork 出 N 个 worker 进程（冷启动 + 重新 mmap 所有分片）。
        #    原代码在「每个梯度累积 micro-step」都调用一次 next(iter_batches(...))，
        #    等于每步都重启一遍数据流水线，GPU 期间只能干等。
        #    这里为 train / val 各缓存一个长生命周期的迭代器，只建一次。
        self._batch_iters = {}

    def _next_batch(self, split="train"):
        """取一个批次；首次调用时才真正创建 DataLoader（之后复用同一个迭代器）。"""
        if split not in self._batch_iters:
            self._batch_iters[split] = iter(self.iter_batches(split=split))
        return next(self._batch_iters[split])

    # 向后兼容：老代码/老脚本可能还在调用 _build_dloader
    _build_dloader = _build_dataloader


    # --------------------- 模型 ---------------------
    # --------------------- 模型构建（像建造摩天大楼） ---------------------
    '''
    主函数 _build_model
            ├─ 参数配置 → 模型选择[scratch->pretrain / resume->ppo] → 设备部署[gpu/cpu]
            ├─ 检查点加载torch.load(ckpt_path) → 参数对齐 → 权重加载self.model.load_state_dict
            ├─ 分词器初始化 → 文本转换基础
            ├─ 训练优化 → 梯度缩放/模型编译
            └─ 分布式包装DDP → 多GPU协同
    '''
    def _build_model(self):
        # [总工程师] 获取建筑蓝图（配置参数）
        arg = self.arg  # 从总控台获取施工参数

        # [材料清单] 准备模型核心参数（建筑材料规格表）
        model_args = dict(
            dim=arg.dim,               # 砖块尺寸：每个单词向量的维度（如512维）
            n_layers=arg.n_layers,     # 楼层总数：Transformer堆叠层数（如12层）
            n_heads=arg.n_heads,       # 施工队数量：注意力机制的头数（如8组）
            n_kv_heads=arg.n_kv_heads, # 特种工程队：键值头的分组数量（如4组）
            vocab_size=arg.vocab_size, # 建材种类：词汇表大小（如50000个词）
            multiple_of=arg.multiple_of, # 尺寸对齐：确保矩阵维度可被整除（如256的倍数）
            max_seq_len=arg.max_seq_len, # 建筑高度：最大处理文本长度（如2048字符）
            dropout=arg.dropout,       # 结构冗余：随机失活比例（如10%的冗余设计）
            cache=arg.ppo,            # 临时仓库：是否启用KV缓存（PPO训练需要）
        )

        # [施工方案选择] 新建或续建工程
        if arg.init_from == "scratch":  # 方案A：新建项目
            if self.master_process:     # 主进程负责打印施工日志
                print("📦 正在打地基：从头初始化新模型")
            gptconf = ModelArgs(**model_args)  # 生成施工图纸
            # 选择建筑类型：普通办公楼 or 带观测塔的特殊建筑
            self.model = (Transformer(gptconf) if not arg.ppo 
                          else TransformerWithValueHead(gptconf))
        else:  # 方案B：续建工程（加载检查点）
            # 🛠️ 原来写死 os.path.join(arg.out_dir, "ckpt.pt")：PPO 阶段存的是 ppo_ckpt.pt，
            #    且 README 里 PPO 阶段的 out_dir 与预训练阶段并不相同，续训必然找不到文件。
            ckpt_path = resolve_ckpt_path(arg.out_dir, arg.ppo, getattr(arg, "ckpt_path", ""))
            if self.master_process:
                print(f"⏯ 正在读取工程存档：{ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=arg.device)  # 加载存档
            self.ckpt_path = ckpt_path  # 供 _build_optimizer 复用同一份存档

            # [结构安全检测] 确保新旧设计参数一致
            for k in ["dim", "n_layers", "n_heads", "n_kv_heads",
                      "vocab_size", "multiple_of", "max_seq_len"]:
                # 🛠️ 兼容两种 checkpoint 格式：
                #    * "model_args"：本仓库 train.py 自己存的（Namespace/dict）
                #    * "config"：llama2.c / 早期脚本存的（data/ckpt.pt 里两种键都有）
                saved = checkpoint.get("model_args", None) or checkpoint.get("config", None)
                assert saved is not None, "checkpoint 里既没有 model_args 也没有 config，无法恢复结构"
                if isinstance(saved, argparse.Namespace):
                    saved = vars(saved)
                if k in saved:
                    model_args[k] = saved[k]   # 强制对齐关键参数
                    # 🛠️ 同时写回 arg：ckpt 里保存的 model_args 决定了下游（sample.py /
                    #    export.py）如何重建模型。若只覆盖局部字典而不回写 arg，
                    #    就会出现“config 写着 dim=288、权重其实是 dim=64”的 checkpoint →
                    #    加载时 size mismatch，训练完根本没法采样/导出。
                    arg.__dict__[k] = saved[k]

            gptconf = ModelArgs(**model_args)  # 重新生成施工图纸
            # 重建建筑框架（与存档结构一致）
            self.model = (Transformer(gptconf) if not arg.ppo 
                          else TransformerWithValueHead(gptconf))

            # [材料装载] 加载之前的建筑材料（模型参数）
            # 🛠️ 用 load_model_state 处理 torch.compile 的 `_orig_mod.` 前缀，
            #    否则 strict=False 会静默丢掉全部权重（详见函数注释）。
            load_model_state(self.model, checkpoint["model"], self.master_process)

            # [恢复工程进度]
            self.iter_num = checkpoint.get("iter_num", 0)          # 当前施工阶段编号
            self.best_val_loss = checkpoint.get("best_val_loss", 1e9) # 历史最佳质量记录

        # [设备运输] 将模型部署到指定设备（GPU/CPU）
        self.model.to(arg.device)  # 如运送至'cuda:0'号施工场地

        # [翻译手册] 初始化分词器（文本→数字的转换字典）
        if arg.tokenizer_path:  # 使用现成的翻译手册
            tokenizer_model = arg.tokenizer_path  # 指定词典路径
        else:  # 自动生成翻译手册
            # 确定词汇表来源：LLAMA2专用词典 or 自定义词典
            query_vocab_size = 0 if arg.vocab_source == "llama2" else arg.vocab_size
            tokenizer_model = get_tokenizer_model_path(query_vocab_size)
        self.enc = Tokenizer(tokenizer_model=tokenizer_model)  # 创建翻译官

        # [安全阀] 梯度缩放器（防止数值下溢）
        self.scaler = torch.cuda.amp.GradScaler(enabled=(arg.dtype == "float16"))
        # 当使用float16时启用，自动调整梯度幅度（类似压力调节器）

        # [流程优化] 模型编译（优化计算路径）
        if arg.compile:  # 开启施工流程优化
            if self.master_process:
                print("⏳ 正在优化施工流程（首次编译约需1分钟）...")
            unoptimized_model = self.model  # 保留原始设计图
            self.model = torch.compile(self.model)  # 生成优化后的流水线
        else:
            # 🛠️ 新增：始终记录“未编译/未包装”的原始模块。
            #    torch.compile 会给参数名加 `_orig_mod.` 前缀（上一轮“续训静默丢权重”的根因），
            #    DDP 还会再包一层。用这个引用去存 ckpt / 导出 / 当 PPO 参考模型最干净。
            unoptimized_model = self.model
        self.unoptimized_model = unoptimized_model

        # [团队协作] 分布式数据并行（组建施工分队）
        if self.ddp:  # 需要多GPU协作
            prefix = "_orig_mod." if arg.compile else ""  # 处理编译后的命名差异
            # 🛠️ 原来写的是 "freqs_cis"（llama 官方仓的名字），本模型里实际叫 freqs_cos /
            #    freqs_sin，等于这个“忽略名单”一直是空操作：DDP 每次前向都会尝试广播这两个
            #    恒定不变的 buffer。这里改成真实名字。
            self.model._ddp_params_and_buffers_to_ignore = {
                prefix + "freqs_cos", prefix + "freqs_sin",
            }
            # 创建分布式施工团队
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        # [总控台访问] 保存原始模型引用
        self.raw_model = self.model.module if self.ddp else self.model
        # 便于直接访问模型核心（绕过分布式包装层）

    '''
# PyTorch DDP工作原理（建筑工地类比）

## 一、工程分组初始化 🏗️
**代码入口**：`dist.init_process_group(backend='nccl')`  
- 每个施工队分配唯一工牌编号（rank）  
- 建立工地专用通信频道（TCP/UDP）  
- 指挥部地址：`MASTER_ADDR='localhost'`  
- 对讲机频道号：`MASTER_PORT='12355'`

## 二、统一图纸分发 📄
**代码实现**：`model = DDP(model, device_ids=[rank])`  
- 指挥部将完整施工图纸（模型参数）同步到各施工队  
- 每个GPU获得完全相同的初始模型副本  
- 通过`device_ids`指定当前GPU专属工作区域  

## 三、分工处理建材 🧱
**代码工具**：`sampler = DistributedSampler(dataset)`  
- 建材库（数据集）被均分成N份（N=施工队数量）  
- 每个施工队领取专属建材批次（数据子集）  
- 防止重复加工（数据去重）  

## 四、协同施工阶段 🔨
1. **正向施工**  
   - 各队独立处理分配的建材（前向传播）  
   - 产出半成品质量检测报告（logits）  

2. **质量检查**  
   - 对照验收标准（标签）计算误差值（loss）  

3. **逆向修正**  
   - 每个施工队独立分析施工缺陷（计算梯度）  
   - 记录整改方案（梯度保存在本地）  

## 五、经验共享大会 📢
**自动执行**：梯度All-Reduce操作  
1. **梯度广播会**  
   - 各队上传整改方案（梯度）  
   - 指挥部计算全局平均值（All-Reduce）  

2. **图纸更新**  
   - 将优化后的参数同步到所有施工队  
   - 确保各队使用最新版本图纸  

## 🚀 DDP协作优势
✅ **效率提升**：N个GPU并行处理N倍数据量  
✅ **质量保障**：通过梯度同步保证模型一致性  
✅ **弹性扩展**：随时增减施工队不影响进度  
    '''
    # --------------------- 优化器 ---------------------
    def _build_optimizer(self):
        # 【总指挥】获取训练配置参数（相当于健身教练的培训手册）
        arg = self.arg  # 从总控制台获取所有配置参数

        # 【组建教练团】创建优化器（相当于聘请健身教练团队）
        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=arg.weight_decay,     # 体重控制系数：防止肌肉过度增长（参数正则化强度）
            learning_rate=arg.learning_rate,   # 学习步长：每次调整动作的幅度（梯度更新步长）
            betas=(arg.beta1, arg.beta2),      # 动量参数：β1=0.9(速度衰减), β2=0.98(方向修正) 
            device_type="cuda" if "cuda" in arg.device else "cpu"  # 训练场地：健身房（GPU）或户外（CPU）
        )
        # 注：该方法内部自动选择AdamW优化器，类似选择专业健身教练

        # 【续训模式】加载历史训练记录（继续之前的健身计划）
        # 🛠️ 原来固定读 out_dir/ckpt.pt，与 _build_model 的查找逻辑不一致；
        #    这里复用同一份解析结果，找不到就跳过（不静默加载错误文件）。
        ckpt_path = getattr(self, "ckpt_path", "") or resolve_ckpt_path(
            arg.out_dir, arg.ppo, getattr(arg, "ckpt_path", "")
        )
        if arg.init_from == "resume" and os.path.exists(ckpt_path):
            try:
                # 打开之前的训练日志（加载检查点文件）
                checkpoint = torch.load(
                    ckpt_path,  # 存档文件路径
                    map_location="cpu"  # 先加载到内存（避免设备不匹配）
                )
                # 教练团读取历史训练数据（加载优化器状态）
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])

                # 主教练播报进度（仅主进程打印）
                if self.master_process:
                    print(f"⤵ 成功载入历史训练记录（动量/学习率等状态）：{ckpt_path}")
            except Exception as e:
                # 当历史记录损坏时的应急方案（如版本不兼容）
                if self.master_process:
                    print(f"⚠️ 旧训练计划失效：{e}，启用全新训练方案")
                # 注：此时优化器会重新初始化，但模型参数仍保持加载状态



    # --------------------- PPO 训练器 ---------------------
    def _build_ppo(self):
        """
        组装 PPO 训练器：策略模型 + 冻结的参考模型 π_ref + 奖励函数。

        🛠️ 本轮新增。原来 train.py 只是把模型换成 TransformerWithValueHead、
           把 ckpt 名字换成 ppo_ckpt.pt，损失依旧是交叉熵 —— ppo.py 的 PPOTrainer
           在整个仓库里没有任何调用点，相当于“PPO 微调”名不副实。
        """
        arg = self.arg
        rollout_size = min(arg.ppo_rollout_size, arg.batch_size)
        if rollout_size != arg.ppo_rollout_size and self.master_process:
            print(f"⚠️ ppo_rollout_size({arg.ppo_rollout_size}) 超过 batch_size({arg.batch_size})，"
                  f"已收敛为 {rollout_size}")
        mini_batch_size = arg.ppo_mini_batch_size or max(1, rollout_size // 8)
        mini_batch_size = min(mini_batch_size, rollout_size)

        # 奖励所要对齐的目标长度 & rollout 生成长度
        target_len = arg.ppo_reward_target_len or arg.target_len
        self.gen_len = arg.ppo_gen_len or min(arg.max_gen_len, arg.max_seq_len - arg.start_len - 1)
        assert self.gen_len > 0, (
            f"生成长度非法：max_gen_len={arg.max_gen_len}, max_seq_len={arg.max_seq_len}, "
            f"start_len={arg.start_len} → 请调小 --start_len 或 --max_gen_len"
        )

        self.ppo_config = PPOConfig(
            seed=1337 + self.seed_offset,
            steps=arg.max_iters,
            learning_rate=arg.ppo_lr if arg.ppo_lr > 0 else arg.learning_rate,
            adap_kl_ctrl=False,
            init_kl_coef=arg.init_kl_coef,
            kl_penalty=arg.ppo_kl_penalty,
            target=arg.target_kl,
            gamma=arg.ppo_gamma,
            lam=arg.ppo_lam,
            cliprange=arg.ppo_cliprange,
            cliprange_value=arg.ppo_cliprange_value,
            vf_coef=arg.ppo_vf_coef,
            batch_size=rollout_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=arg.ppo_grad_accum_steps,
            ppo_epochs=arg.ppo_epochs,
            max_grad_norm=arg.grad_clip if arg.grad_clip > 0 else None,
            early_stopping=arg.ppo_early_stopping,
            target_kl=arg.target_kl,
            whiten_rewards=arg.ppo_whiten_rewards,
        )

        # 参考策略 π_ref：本阶段开始时的策略副本，冻结参数、只做前向
        ref_model = deepcopy(self.unoptimized_model)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()
        ref_model.to(arg.device)
        # 🛠️ 参考模型永远不带 KV cache 前向，把它的 cache 张量丢掉，省一半显存
        #    （Attention 里 cache_k/cache_v 是普通属性、不是 buffer，不会被 .to() 移动，
        #     所以这里显式置空最稳妥）
        if getattr(ref_model, "params", None) is not None and ref_model.params.cache:
            for layer in ref_model.layers:
                layer.attention.cache = False
                layer.attention.cache_k = None
                layer.attention.cache_v = None
        self.ref_model = ref_model

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.enc,
            optimizer=self.optimizer,
            scaler=self.scaler,
            is_ddp=self.ddp,
        )
        # 恢复 KL 系数（续训时不要让 KL 惩罚从头再来）
        ckpt_path = getattr(self, "ckpt_path", "")
        if arg.init_from == "resume" and ckpt_path and os.path.exists(ckpt_path):
            try:
                saved = torch.load(ckpt_path, map_location="cpu")
                if "kl_coef" in saved:
                    self.ppo_trainer.kl_ctl.value = float(saved["kl_coef"])
                    if self.master_process:
                        print(f"⤵ 恢复 KL 系数：{self.ppo_trainer.kl_ctl.value:.4f}")
            except Exception as e:
                if self.master_process:
                    print(f"⚠️ KL 系数恢复失败：{e}")

        # PPO 阶段的学习率（通常比预训练低一个量级）
        lr = self.ppo_config.learning_rate
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.reward_fn = build_reward_fn(arg.ppo_reward, target_len, self.enc, sampler=None)
        # 目标长度采样：给了明确的 --ppo_reward_target_len 就用固定值，否则在
        # [start_len, target_len] 内随机采样（ppo.py 自带 LengthSampler 是左闭右开区间），
        # 并且把上界收敛到本次 rollout 真正能生成的长度，避免奖励恒为同一个极小值。
        if arg.ppo_reward_target_len <= 0 and "length" in self.reward_fn.kinds:
            lo = max(1, min(arg.start_len, self.gen_len))
            hi = min(max(arg.target_len, lo + 1), self.gen_len)
            self.reward_fn.sampler = LengthSampler(min_value=lo, max_value=hi + 1)
        # 其余情况（给了固定目标长度、或使用 completion 奖励）不需要采样
        if self.master_process:
            sampler_desc = (
                f"每条随机采样目标长度 [{self.reward_fn.sampler.values[0]}, "
                f"{self.reward_fn.sampler.values[-1]}]"
                if self.reward_fn.sampler is not None else f"固定目标长度 {target_len}"
            )
            print(
                "📣 PPO 配置："
                f"rollout={rollout_size} × {arg.ppo_epochs} epoch（mini-batch={mini_batch_size}）、"
                f"prompt={arg.start_len} tok、生成={self.gen_len} tok、"
                f"奖励={arg.ppo_reward}（{sampler_desc}）、KL 系数 {arg.init_kl_coef}、lr {lr:g}"
            )

    # -----------------------------------------------------------------
    # 👉 4. 训练循环
        '''
        训练主循环
        ├─ 学习率调整 → 参数更新 → 定期评估
        ├─ 梯度累积 → 反向传播 → 梯度裁剪
        ├─ 优化器更新 → 日志记录 → 迭代计数
        └─ 终止条件判断 → 清理资源

        '''
    # -----------------------------------------------------------------
    def train(self):
        # 【工厂总控】获取训练配置参数（相当于生产车间的操作手册）
        arg = self.arg
        # 🛠️ 新增分发：--ppo 时走真正的强化学习循环（原来只换了模型头，损失仍是交叉熵）
        if arg.ppo:
            return self.train_ppo()
        # 🛠️ 用 _next_batch 复用同一个长生命周期迭代器（原代码每次 next(iter_batches(...))
        #    都会新建 DataLoader / 重启 worker 进程，见 _build_dataloader 里的说明）
        next_batch = self._next_batch  # 原材料供应管道（数据批次生成器）
        X, Y = next_batch("train")  # 首次获取原料（训练数据的输入X和标签Y）

        # 【产能计算】计算每次迭代处理的总数据量（相当于工厂的日产量）
        '''
        # 假设 accumulation_steps=4, batch_size=32
        总有效批次 = 32 * 4 = 128
        每个微批次计算梯度后不立即更新，而是累积4次后再统一更新
        相当于用较小显存实现大批量训练
        '''
        tokens_per_iter = (arg.gradient_accumulation_steps *  # 累积次数（分批次加工）
                          self.ddp_world_size *               # 分布式节点数GPU数（分厂数量）
                          arg.batch_size *                    # 单批次容量（单条生产线产能）
                          arg.max_seq_len)                    # 序列长度（原料尺寸）
        if self.master_process:  # 主控台显示生产信息
            print(f"每次迭代处理 token 数: {tokens_per_iter:,}")  # 如显示"每次处理 524,288 tokens"

        # 【计时系统】初始化训练计时器和关键指标
        t0 = time.time()                    # 启动秒表
        iter_num = getattr(self, "iter_num", 0)        # 当前生产批次编号（从0开始计数）
        best_val_loss = getattr(self, "best_val_loss", 1e9)  # 历史最佳质检成绩（初始设为极大值）

        # === 核心训练循环（生产线主流程） ===
        while True:
            # --- 动态学习率调节（类似健身时的强度调整）---
            lr = get_lr(iter_num, arg, self.target_iters) if arg.decay_lr else arg.learning_rate  # 获取当前学习率
            for pg in self.optimizer.param_groups:  # 更新所有参数组的学习率[for DDP不同GPU分工]
                pg["lr"] = lr  # 相当于调整健身动作的幅度

            # --- 质量检测与存档（定期产品质检）---
            if iter_num % arg.eval_interval == 0 and self.master_process:  # 每N次迭代检测
                losses = estimate_loss(self.model, next_batch, self.ctx, arg)  # 质量评估
                print(f"step {iter_num}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")

                # 当检测到更优模型时存档（破纪录存档）
                if losses["val"] < best_val_loss or arg.always_save_checkpoint:
                    best_val_loss = losses["val"]  # 更新最佳记录
                    if iter_num > 0:  # 首次迭代不保存
                        ckpt = {  # 存档内容
                            # 🛠️ 存“未编译”的模块：否则 --compile 会把参数名变成
                            #    `_orig_mod.xxx` 写进 ckpt，续训/采样时都得额外剥前缀
                            "model": self.unoptimized_model.state_dict(),  # 生产线当前参数
                            "optimizer": self.optimizer.state_dict(),  # 优化器状态（动量等）
                            # 🛠️ 原来是 asdict(arg)：arg 是 argparse.Namespace 不是 dataclass，
                            #    这行会抛 TypeError（asdict() should be called on dataclass
                            #    instances），也就是第一次评估保存 checkpoint 时必崩。
                            "model_args": checkpoint_args(arg, self.unoptimized_model),  # 生产配置参数（以模型真实结构为准）
                            "iter_num": iter_num,                      # 当前批次编号
                            "best_val_loss": best_val_loss,            # 最佳质检成绩
                        }
                        fname = "ppo_ckpt.pt" if arg.ppo else "ckpt.pt"  # 特殊生产工艺存档
                        torch.save(ckpt, os.path.join(arg.out_dir, fname))  # 保存到指定目录
                        model_export(self.raw_model, os.path.join(arg.out_dir, "model.bin"), version=0)  # 导出成品

                if iter_num == 0 and arg.eval_only:  # 仅评估模式时提前退出
                    break

            # === 核心生产流程（前向传播+反向传播）===
            # 【分步加工】梯度累积（类似分期完成大额支付）
            for micro in range(arg.gradient_accumulation_steps):
                # 🛠️ 原来直接改 DDP 内部属性 self.model.require_backward_grad_sync：
                #    这是私有实现细节，在 torch.compile / 新版 torch 下随时可能失效或静默
                #    不生效（多卡时梯度同步次数错误，等价于 batch 变大/变小）。
                #    官方推荐用 no_sync() 上下文，语义一致且稳定。
                sync_ctx = (
                    nullcontext() if (not self.ddp) or micro == arg.gradient_accumulation_steps - 1
                    else self.model.no_sync()
                )
                with sync_ctx:

                    # 【原料加工】前向计算
                    with self.ctx:  # 自动混合精度上下文（精密加工车间）
                        _, loss = self.model(X, Y)[-2:]    # 获取预测结果和损失值
                        loss = loss / arg.gradient_accumulation_steps  # 分摊损失（分期付款）

                    X, Y = next_batch("train")  # 预取下一批原料（流水线持续供料）
                    self.scaler.scale(loss).backward()       # 反向传播计算梯度（质量分析报告）

            # 【工艺优化】梯度处理与参数更新
            if arg.grad_clip != 0.0:  # 梯度裁剪（防止过激调整）
                self.scaler.unscale_(self.optimizer)       # 解除缩放（恢复原始梯度值）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), arg.grad_clip)  # 裁剪过大的梯度
            self.scaler.step(self.optimizer)    # 执行参数更新（根据分析调整生产线）
            self.scaler.update()                # 更新缩放器状态（校准测量仪器）
            self.optimizer.zero_grad(set_to_none=True)  # 清空梯度（为下一批生产准备）

            # === 生产日志记录 ===
            t1 = time.time()  # 记录本批次结束时间
            if iter_num % arg.log_interval == 0 and self.master_process:  # 定期打印日志
                lossf = loss.item() * arg.gradient_accumulation_steps  # 还原实际损失值
                print(f"{iter_num} | 损失 {lossf:.4f} | 学习率 {lr:.2e} | 耗时 {(t1 - t0)*1000:.2f} ms")
            t0 = t1  # 重置计时器
            iter_num += 1  # 更新批次计数器

            if iter_num > self.target_iters:  # 达到目标步数时停机
                break

        # === 生产收尾工作 ===
        if self.ddp:  # 分布式训练时关闭协作通道
            destroy_process_group()  # 关闭分布式进程组（各分厂通讯断开）
        if self.master_process:  # 主控台最终报告
            print("🏁 训练完成！最新模型与日志已保存在", arg.out_dir)


    # -----------------------------------------------------------------
    # 👉 4.5 PPO 训练循环（真正的强化学习微调：rollout → 奖励 → GAE → clip 更新）
    # -----------------------------------------------------------------
    def _rollout(self):
        """
        采样一批 (prompt, response)：
          prompt 取自训练集里真实故事的开头（前 start_len 个 token），
          response 由当前策略模型用 KV cache 生成，
          score 由 self.reward_fn 打分（长度 / 结尾完整性 / 两者相加）。
        """
        arg = self.arg
        if arg.ppo_prompt_file:
            prompts = self._prompts_from_file(b=arg.ppo_rollout_size)
        else:
            X, _ = self._next_batch("train")        # [B, T] 真实故事 token
            b = min(arg.ppo_rollout_size, arg.batch_size, X.shape[0])
            # 🛠️ prompt 统一取 start_len 个 token：批生成时不会出现 padding，
            #    KV cache 里也就不会混入 pad token（见 ppo/generate 相关注释）。
            prompts = [X[i, :arg.start_len].tolist() for i in range(b)]

        self.model.eval()                            # 采样阶段：关闭 dropout、走 eval 分支
        with torch.no_grad():
            eos_id = self.enc.eos_id if self.enc.eos_id is not None else self.enc.bos_id
            full_seqs = self.model.generate_withcache(
                prompts,
                max_gen_len=self.gen_len,
                temperature=arg.ppo_temperature,
                top_k=arg.ppo_top_k,
                top_p=arg.ppo_top_p,
                eos_id=eos_id,
                echo=False,                          # 只取新生成的 response（prompt 不返回）
                device=arg.device,
            )

        queries, responses, scores = [], [], []
        for prompt, seq in zip(prompts, full_seqs):
            resp = list(seq)                         # echo=False → seq 已经是纯 response
            if not resp:                             # 兜底：至少一个 token 才能算 logprob
                resp = [self.enc.bos_id]
            queries.append(torch.tensor(prompt, dtype=torch.long))
            responses.append(torch.tensor(resp, dtype=torch.long))
            scores.append(torch.tensor([self.reward_fn(resp)], dtype=torch.float32))
        return queries, responses, scores

    def _ppo_eval_batch(self, split="val"):
        """
        PPO 质量监控用的**小批量**数据流。

        🛠️ 为什么单独建一条：训练用的 batch_size 默认 128、max_seq_len 默认 256，
           一次评估前向（128×256=32768 token）在 2 核 CPU 上要 ~3 秒，
           而监控只需要看趋势 —— 16×256 就够。实测 B=16/T=48 的前向只要 ~25ms，
           规模差 40 倍以上再叠加注意力的 T² 项，3 秒完全对得上，不是 bug。
        """
        if not hasattr(self, "_ppo_eval_iters"):
            self._ppo_eval_iters = {}
        if split not in self._ppo_eval_iters:
            arg = self.arg
            iters = partial(
                Task.iter_batches,
                batch_size=arg.ppo_eval_batch_size,
                max_seq_len=min(arg.max_seq_len, 256),
                vocab_size=arg.vocab_size,
                vocab_source=arg.vocab_source,
                device=arg.device,
                num_workers=0,          # 评估批次小，起 worker 不划算
            )
            self._ppo_eval_iters[split] = iter(iters(split=split))
        return next(self._ppo_eval_iters[split])

    def _prompts_from_file(self, b: int):
        """
        从 --ppo_prompt_file 读取 prompt：每行一条故事开头，编码后截断到 start_len 个 token。
        返回等长的 token 列表（等长 → 批生成无 padding，KV cache 干净）。
        """
        arg = self.arg
        if getattr(self, "_prompt_pool", None) is None:
            with open(arg.ppo_prompt_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            pool = []
            for ln in lines:
                ids = self.enc.encode(ln, bos=True, eos=False)
                if len(ids) < arg.start_len:  # 太短的 prompt 直接跳过（避免凑长度引入噪声）
                    continue
                pool.append(ids[:arg.start_len])
            assert pool, (
                f"{arg.ppo_prompt_file} 里没有长度 ≥ start_len({arg.start_len}) 的 prompt，"
                f"请调小 --start_len 或补充更长的数据"
            )
            self._prompt_pool = pool
            if self.master_process:
                print(f"📄 从 {arg.ppo_prompt_file} 载入 {len(pool)} 条 prompt（每条截断到 {arg.start_len} token）")
        pool = self._prompt_pool
        idx = np.random.randint(0, len(pool), size=b)  # 每步随机采样一批
        return [list(pool[i]) for i in idx]

    def _ppo_checkpoint(self, iter_num: int, best_val_loss: float, stats: dict):
        """保存 PPO checkpoint（含 KL 系数，续训时 KL 惩罚不会从头再来）。"""
        arg = self.arg
        path = os.path.join(arg.out_dir, "ppo_ckpt.pt")
        torch.save(
            {
                "model": self.unoptimized_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_args": checkpoint_args(arg, self.unoptimized_model),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "kl_coef": float(self.ppo_trainer.kl_ctl.value),
                "ppo_stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            },
            path,
        )
        return path

    def train_ppo(self):
        """
        PPO 主循环：每个 step 做一次「采样 rollout → 计算 KL/GAE → clip 更新」。

        🛠️ 本轮新增，把 ppo.py 的 PPOTrainer 真正接进 train.py。
           数据流：训练集取 prompt → 策略生成 response → 奖励打分 → PPOTrainer.step()
                  （内部完成 π_ref 前向、KL 惩罚、GAE、PPO-clip 损失、梯度累积与更新）
        """
        arg = self.arg
        # DDP 下各 rank 各自采样数据（数据并行），梯度同步交给 PPOTrainer.step() 内部处理
        iter_num = getattr(self, "iter_num", 0)
        best_val_loss = getattr(self, "best_val_loss", 1e9)

        # 🛠️ 目标步数兜底：resume 时 ckpt 里的 iter_num 常常已经 ≥ max_iters（例如接着
        #    iter 98000 的模型做 PPO，而 max_iters 只有几千），若按“总步数”解释就会一步不跑。
        #    PPO 是新一轮微调，这里默认按“本阶段再训 max_iters 步”处理，并明确打印出来。
        target_iters = self.target_iters
        if arg.init_from == "resume" and iter_num >= target_iters and not arg.iters_from_now:
            target_iters = iter_num + arg.max_iters
            if self.master_process:
                print(f"ℹ️ 检测到 ckpt 已训练 {iter_num} 步 ≥ max_iters({arg.max_iters})，"
                      f"本轮按「再训 {arg.max_iters} 步」处理 → 目标 {target_iters} 步"
                      f"（如需其它语义请显式使用 --iters_from_now）")
        if self.master_process:
            print(f"🚀 开始 PPO 训练：{iter_num} → {target_iters} 步，"
                  f"每个 step 采样 {self.ppo_config.batch_size} 条")

        t0 = time.time()
        stats = {}
        while True:
            # --- 学习率：PPO 阶段通常固定 lr（--decay_lr=False）---
            lr = get_lr(iter_num, arg, target_iters) if arg.decay_lr else self.ppo_config.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # 🛠️ 学习率为 0 时给一次明确警告：这会让 PPO 全程不更新参数（比报错更隐蔽）
            if lr == 0.0 and not getattr(self, "_warned_zero_lr", False):
                self._warned_zero_lr = True
                if self.master_process:
                    print("⚠️ 当前学习率为 0（decay_lr 已把 lr 衰减到 min_lr），PPO 不会更新参数！"
                          "请设置 --min_lr>0、--decay_lr=False 或加大 --max_iters/使用 --iters_from_now")


            # --- ① 采样 rollout（prompt + response + reward）---
            queries, responses, scores = self._rollout()

            # --- ② 一次 PPO 更新（内部：π_ref 前向 → KL 惩罚 → GAE → clip 损失 → 反向）---
            stats = self.ppo_trainer.step(queries=queries, responses=responses, scores=scores)

            # --- ③ 日志 ---
            t1 = time.time()
            if iter_num % arg.log_interval == 0 and self.master_process:
                rewards = [float(s.mean()) for s in scores]
                print(
                    f"[PPO] step {iter_num} | 奖励 {stats['ppo/reward_all']:+.3f} "
                    f"(本批 {np.mean(rewards):+.3f} ± {np.std(rewards):.3f}) "
                    f"| KL_ref {stats['ppo/kl_ref']:+.4f} | policykl {stats.get('ppo/policy/policykl', float('nan')):+.4f} "
                    f"| 熵 {stats.get('ppo/policy/entropy', float('nan')):.3f} "
                    f"| clipfrac {stats.get('ppo/policy/clipfrac', float('nan')):.3f} "
                    f"| loss {stats.get('ppo/loss/total', float('nan')):+.4f} "
                    f"| 响应长度 {np.mean([len(r) for r in responses]):.1f} "
                    f"| lr {lr:.2e} | {(t1 - t0) * 1000:.0f} ms"
                )
                # 让 RL 的效果肉眼可见：每隔一段时间打印一条真实生成结果
                if arg.ppo_debug:
                    sample = self.enc.decode(responses[0].tolist()).strip()
                    print(f"       └─ 样例（奖励 {rewards[0]:+.3f}）：{sample[:120]}")
            t0 = t1
            iter_num += 1

            # --- ④ 质量监控：用验证集 CE 损失看语言能力有没有被 RL 练坏 ---
            # （RL 微调的常见事故就是奖励涨了、但模型不会说人话了；这里给一个显式指标）
            if arg.eval_iters > 0 and iter_num % arg.eval_interval == 0 and self.master_process:
                try:
                    n_eval = min(arg.eval_iters, 10)   # PPO 阶段不必跑满 eval_iters，够看趋势即可
                    old_eval_iters, arg.eval_iters = arg.eval_iters, n_eval
                    try:
                        losses = estimate_loss(self.model, self._ppo_eval_batch, self.ctx, arg)
                        print(f"  📊 CE 损失（质量监控）：train {losses['train']:.4f} | val {losses['val']:.4f}")
                    finally:
                        arg.eval_iters = old_eval_iters
                except Exception as e:
                    if not getattr(self, "_warned_ppo_eval", False):
                        self._warned_ppo_eval = True
                        print(f"  ⚠️ PPO 质量监控已跳过（{type(e).__name__}: {e}）")

            # --- ⑤ 存档（PPO 阶段按 eval_interval 定期保存最新模型）---
            if (iter_num % arg.eval_interval == 0 or iter_num >= target_iters) and self.master_process:
                path = self._ppo_checkpoint(iter_num, best_val_loss, stats)
                print(f"💾 已保存 {path}（含 KL 系数 {self.ppo_trainer.kl_ctl.value:.4f}）")

            if iter_num >= target_iters:
                break

        if self.ddp:
            destroy_process_group()
        if self.master_process:
            print("🏁 PPO 训练完成！模型与日志已保存在", arg.out_dir)


# -----------------------------------------------------------------------------
# 👉 5. 入口
# -----------------------------------------------------------------------------
def main():
    arg = TrainConfig()   # 如需修改超参，可在此处或命令行解析后覆盖
    trainer = TinyStoriesTrainer(arg)
    trainer.train()


if __name__ == "__main__":
    main()