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
from dataclasses import dataclass, asdict

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
    lr_group.add_argument("--decay_lr", action="store_false",default=True,  # ✅ store_false对应默认True
                    help="禁用学习率衰减（默认启用）")  
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
    device_group.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], 
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

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 👉 2. 工具函数：学习率调度 & 评估
# -----------------------------------------------------------------------------
import math

def get_lr(it: int, arg: TrainConfig) -> float:
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
    if it < arg.warmup_iters:
        return arg.learning_rate * it / arg.warmup_iters
    
    # ================ 训练终止：保护机制 ================
    # 当超过最大迭代步数时保持最小学习率
    # 作用：防止后续计算产生意外波动
    if it > arg.max_iters:
        return arg.min_lr
    
    # ============== 训练中：余弦衰减阶段（核心计算） ==============
    # 计算衰减进度：[0,1]区间，0表示刚结束预热，1表示训练结束
    # 示例：当 warmup=1000, max=10000, it=2000 时：
    # decay_ratio = (2000-1000)/(10000-1000) ≈ 0.111
    decay_ratio = (it - arg.warmup_iters) / (arg.max_iters - arg.warmup_iters)
    
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
@torch.no_grad() # ← 评估时不计算梯度，节省显存/速度快
@torch.no_grad()  # 魔法开关：关闭自动求导功能，就像考试时不需要记笔记
def estimate_loss(model, iter_batches, ctx, arg: TrainConfig):
    """评估模型在训练集和验证集上的平均损失（相当于模拟考试）"""
    # 结果字典：存放训练集和验证集的平均损失
    out = {}
    
    # 切换模型为考试模式：关闭 dropout / LayerNorm 统计等训练专用功能，类似关闭手机专心考试
    model.eval()  
    
    # 同时在训练集和验证集上测试，好比同时做课堂练习和期末考试
    for split in ["train", "val"]:  
        # 获取该数据集的数据加载器（就像拿到考卷）
        batch_iter = iter_batches(split=split)  
        # 创建存放多次考试结果的成绩单（arg.eval_iters次测试取平均）
        losses = torch.zeros(arg.eval_iters)  
        
        # 进行多次小测试取平均（避免单次考试的偶然性）
        for k in range(arg.eval_iters):
            # 获取一个批次的题目和答案（X是题目，Y是标准答案）
            X, Y = next(batch_iter)  
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
            exit()
            
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
    def _build_dloader(self):
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
            num_workers=0,                  # 分拣员数量（0表示不用多线程）
        )


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
            ckpt_path = os.path.join(arg.out_dir, "ckpt.pt")  # 找到之前的工程存档
            if self.master_process:
                print(f"⏯ 正在读取工程存档：{ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=arg.device)  # 加载存档

            # [结构安全检测] 确保新旧设计参数一致
            for k in ["dim", "n_layers", "n_heads", "n_kv_heads",
                      "vocab_size", "multiple_of", "max_seq_len"]:
                model_args[k] = checkpoint["model_args"][k]  # 强制对齐关键参数

            gptconf = ModelArgs(**model_args)  # 重新生成施工图纸
            # 重建建筑框架（与存档结构一致）
            self.model = (Transformer(gptconf) if not arg.ppo 
                          else TransformerWithValueHead(gptconf))

            # [材料装载] 加载之前的建筑材料（模型参数）
            badkeys = self.model.load_state_dict(checkpoint["model"], strict=False)
            if self.master_process:
                # 报告材料差异：正常情况只有 freqs_cis 不匹配
                print(f"✓ 材料装载完成，差异报告：{badkeys}")  

            # [恢复工程进度]
            self.iter_num = checkpoint["iter_num"]          # 当前施工阶段编号
            self.best_val_loss = checkpoint["best_val_loss"] # 历史最佳质量记录

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

        # [团队协作] 分布式数据并行（组建施工分队）
        if self.ddp:  # 需要多GPU协作
            prefix = "_orig_mod." if arg.compile else ""  # 处理编译后的命名差异
            # 指定需要忽略的共享参数（如旋转位置编码）
            self.model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
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
        if arg.init_from == "resume" and os.path.exists(os.path.join(arg.out_dir, "ckpt.pt")):
            try:
                # 打开之前的训练日志（加载检查点文件）
                checkpoint = torch.load(
                    os.path.join(arg.out_dir, "ckpt.pt"),  # 存档文件路径
                    map_location="cpu"  # 先加载到内存（避免设备不匹配）
                )
                # 教练团读取历史训练数据（加载优化器状态）
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                # 主教练播报进度（仅主进程打印）
                if self.master_process:
                    print("⤵ 成功载入历史训练记录（动量/学习率等状态）")
            except Exception as e:
                # 当历史记录损坏时的应急方案（如版本不兼容）
                if self.master_process:
                    print(f"⚠️ 旧训练计划失效：{e}，启用全新训练方案")
                # 注：此时优化器会重新初始化，但模型参数仍保持加载状态



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
        iter_batches = self.iter_batches  # 原材料供应管道（数据批次生成器）
        X, Y = next(iter_batches(split="train"))  # 首次获取原料（训练数据的输入X和标签Y）

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
            lr = get_lr(iter_num, arg) if arg.decay_lr else arg.learning_rate  # 获取当前学习率
            for pg in self.optimizer.param_groups:  # 更新所有参数组的学习率[for DDP不同GPU分工]
                pg["lr"] = lr  # 相当于调整健身动作的幅度

            # --- 质量检测与存档（定期产品质检）---
            if iter_num % arg.eval_interval == 0 and self.master_process:  # 每N次迭代检测
                losses = estimate_loss(self.model, iter_batches, self.ctx, arg)  # 质量评估
                print(f"step {iter_num}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")

                # 当检测到更优模型时存档（破纪录存档）
                if losses["val"] < best_val_loss or arg.always_save_checkpoint:
                    best_val_loss = losses["val"]  # 更新最佳记录
                    if iter_num > 0:  # 首次迭代不保存
                        ckpt = {  # 存档内容
                            "model": self.raw_model.state_dict(),      # 生产线当前参数
                            "optimizer": self.optimizer.state_dict(),  # 优化器状态（动量等）
                            "model_args": asdict(arg),                 # 生产配置参数
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
                if self.ddp:  # 分布式训练时同步控制
                    # 仅在最后一次微批次时同步梯度（类似分厂最后统一汇总）
                    self.model.require_backward_grad_sync = (micro == arg.gradient_accumulation_steps - 1)

                # 【原料加工】前向计算
                with self.ctx:  # 自动混合精度上下文（精密加工车间）
                    _, loss = self.model(X, Y)[-2:]    # 获取预测结果和损失值
                    loss = loss / arg.gradient_accumulation_steps  # 分摊损失（分期付款）

                X, Y = next(iter_batches(split="train"))  # 预取下一批原料（流水线持续供料）
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

            if iter_num > arg.max_iters:  # 达到最大训练次数时停机
                break

        # === 生产收尾工作 ===
        if self.ddp:  # 分布式训练时关闭协作通道
            destroy_process_group()  # 关闭分布式进程组（各分厂通讯断开）
        if self.master_process:  # 主控台最终报告
            print("🏁 训练完成！最新模型与日志已保存在", arg.out_dir)



# -----------------------------------------------------------------------------
# 👉 5. 入口
# -----------------------------------------------------------------------------
def main():
    arg = TrainConfig()   # 如需修改超参，可在此处或命令行解析后覆盖
    trainer = TinyStoriesTrainer(arg)
    trainer.train()


if __name__ == "__main__":
    main()