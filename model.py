import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # 默认的Llama 7B模型超参数配置
    dim: int = 4096  # 模型的嵌入维度，每个词汇对应的向量长度（通常在Transformer模型中较大）
    n_layers: int = 32  # Transformer模型的层数，也就是Transformer块的数量
    n_heads: int = 32  # 每一层的注意力头数（多头注意力机制）
    n_kv_heads: Optional[int] = None  # 键（Key）和值（Value）的注意力头数（如果为None，则使用n_heads的数量）
    vocab_size: int = 32000  # 词汇表的大小，指模型能处理的不同词汇的数量
    hidden_dim: Optional[int] = None  # 隐藏层维度（默认为None）
    multiple_of: int = 256  # MLP（多层感知机）隐藏层的维度将是此值的倍数
    norm_eps: float = 1e-5  # 归一化时添加的小值，防止出现除以零的错误
    max_seq_len: int = 2048  # 最大序列长度，模型输入的最大token数
    max_batch_size: int = 32  # 推理时的最大批次大小
    dropout: float = 0.0  # Dropout比率，用于防止过拟合
    cache: bool = False  # 是否启用缓存机制

# RMSNorm归一化
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  # 防止计算除零时的微小值
        self.weight = nn.Parameter(torch.ones(dim))  # 用于学习的参数，初始化为1

    def _norm(self, x):
        # RMSNorm的标准化过程：每个输入元素按其平方均值的倒数进行缩放
        # 这里计算的是 (x * 1 / sqrt(mean(x^2) + epsilon))，epsilon防止除零错误
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 标准化输入x并返回加权输出
        output = self._norm(x.float()).type_as(x)  # 转换为float进行计算，然后恢复类型
        return output * self.weight  # 加权输出


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 预计算频率值，用于旋转位置编码（RoPE）
    # 频率的计算公式：freqs = 1 / (theta ^ (arange(0, dim, 2) / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 计算每个维度的频率
    t = torch.arange(end, device=freqs.device)  # 创建一个长度为end的张量（对应位置）

    freqs = torch.outer(t, freqs).float()  # 外积计算频率矩阵
    freqs_cos = torch.cos(freqs)  # 计算频率的余弦部分
    freqs_sin = torch.sin(freqs)  # 计算频率的正弦部分
    return freqs_cos, freqs_sin  # 返回余弦和正弦部分


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 调整频率矩阵的形状，使其可以广播到输入x的形状
    ndim = x.ndim  # 获取输入x的维度数
    assert 0 <= 1 < ndim  # 确保x有足够的维度
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # 检查频率矩阵的形状与x的最后两个维度匹配

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # 根据x的维度创建新的形状
    return freqs_cis.view(shape)  # 返回调整后的频率矩阵


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 应用旋转位置编码（RoPE）到查询和键的向量中
    # 先将查询（xq）和键（xk）张量按复数形式拆解（实部和虚部）
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 调整频率矩阵的形状以便广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 对查询和键进行旋转操作，分别处理实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # 查询的实部
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # 查询的虚部
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # 键的实部
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # 键的虚部

    # 将最后两个维度展平
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)  # 返回处理后的查询和键


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 重复键（key）和值（value）n_rep次
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # 在最后一个维度上添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 扩展该维度，重复n_rep次
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 将扩展后的维度进行reshape
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 初始化注意力机制的参数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0  # 确保键和值头数可以被头数整除
        
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size  # 每个并行的注意力头数
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size  # 每个并行的键值头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 计算每个键值头重复的次数
        self.head_dim = args.dim // args.n_heads  # 每个头的维度

        # 定义查询、键、值的线性变换层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # 输出线性层

        # Dropout操作，用于防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.cache = args.cache  # 是否启用缓存

        # 如果启用了缓存，初始化缓存的键和值
        if args.cache:
            self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ))
            self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ))

        # 检查是否可以使用flash attention（加速版本）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache: Optional[bool] = False,
    ):
        bsz, seqlen, _ = x.shape  # 获取批次大小和序列长度

        # 计算查询、键、值
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码（RoPE）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 如果启用了缓存，则保存当前的键和值，以备下次使用
        if cache and not self.training:
            assert self.cache, "cache must be enabled"
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            xk = self.cache_k[:bsz, : start_pos + seqlen]
            xv = self.cache_v[:bsz, : start_pos + seqlen]

        # 扩展键和值，使得每个查询都有多个键和值（多查询注意力）
        xk = repeat_kv(xk, self.n_rep)  # 扩展键
        xv = repeat_kv(xv, self.n_rep)  # 扩展值

        # 将头部维度转换为批次维度
        xq = xq.transpose(1, 2)  # 将查询张量的维度调整为(batch_size, n_local_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 使用flash attention（快速实现）或者慢速实现
        if self.flash:
            # 使用flash attention（PyTorch中优化的点积注意力）
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        else:
            # 手动实现注意力计算
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            scores = scores + mask  # 使用mask来排除无效位置
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # 使用softmax计算注意力权重
            scores = self.attn_dropout(scores)  # 应用dropout
            output = torch.matmul(scores, xv)  # 根据注意力权重计算输出

        # 恢复时间维度并将所有头的输出拼接
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最后的线性变换，将所有头的输出合并并送入线性层
        output = self.wo(output)
        output = self.resid_dropout(output)  # 最后的dropout
        return output  # 返回注意力计算后的输出


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果hidden_dim未提供，则根据dim计算一个默认值
        # hidden_dim通常是dim的4倍，然后根据multiple_of进行调整，确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim  # 默认为dim的4倍
            hidden_dim = int(2 * hidden_dim / 3)  # 稍微缩小一下维度
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 调整为multiple_of的倍数
        # 定义MLP的层
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 输入到隐藏层的线性变换
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 隐藏层到输出的线性变换
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 额外的线性层，用于调整输出
        self.dropout = nn.Dropout(dropout)  # Dropout操作，用于防止过拟合

    def forward(self, x):
        # MLP的前向传播过程：先通过w1线性变换，再使用ReLU激活函数（silu），再通过w3，最后通过w2输出
        # 通过w1线性变换后再激活，然后乘上w3的输出，最后通过w2层进行输出
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))  # 计算并返回MLP的结果


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # 定义Transformer块中的所有组件，包括注意力层和前馈层
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        self.attention = Attention(args)  # 注意力层
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )  # 前馈神经网络层
        self.layer_id = layer_id  # 当前层的ID
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 注意力层的归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 前馈层的归一化

    def forward(self, x, start_pos, freqs_cos, freqs_sin, mask, cache):
        # Transformer块的前向传播：通过注意力层和前馈层计算输出
        # 先将输入加上注意力层的输出（残差连接）
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cos, freqs_sin, mask, cache)
        # 再将结果加上前馈层的输出（再次使用残差连接）
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers  # Transformer的层数

        # 词嵌入层：将词汇表中的每个token映射到一个向量空间中
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)  # Dropout层，用于防止过拟合
        self.layers = torch.nn.ModuleList()  # 创建一个空的层列表，存放所有TransformerBlock
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))  # 添加每一层的TransformerBlock
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 对最后的输出做归一化
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 输出层，将嵌入空间的输出转换回词汇空间

        # 共享嵌入层和输出层的权重（为了减少参数量）
        self.tok_embeddings.weight = self.output.weight  # 参考论文的权重共享技术

        # 预计算RoPE相对位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册缓存变量
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对某些残差层进行特殊的初始化，参考GPT-2的初始化方式
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))  # 对权重进行正态分布初始化

    def _init_weights(self, module):
        # 权重初始化函数：根据不同类型的模块进行不同初始化
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 对线性层进行正态分布初始化
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 对bias进行零初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 对词嵌入层进行初始化

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, start_pos: int = 0, cache: bool = False) -> torch.Tensor:
        # Transformer模型的前向传播
        _bsz, seqlen = tokens.shape  # 获取批次大小和序列长度
        h = self.tok_embeddings(tokens)  # 获取tokens对应的嵌入向量
        h = self.dropout(h)  # 应用dropout

        # 获取对应的频率余弦和正弦部分，用于相对位置编码（RoPE）
        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]

        # 创建mask矩阵，用于屏蔽掉当前时间步之前的token（自回归）
        mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )
        mask = torch.triu(mask, diagonal=1)  # 上三角矩阵，保证当前位置只能看到之前的token
        mask = torch.hstack([
            torch.zeros((seqlen, start_pos), device=tokens.device),
            mask
        ]).type_as(h)  # 拼接前置部分，形成完整的mask

        # 将输入通过每一层TransformerBlock进行前向传播
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cos, freqs_sin, mask, cache)
        h = self.norm(h)  # 最后的归一化

        if targets is not None:
            # 如果提供了目标，则计算损失
            logits = self.output(h)  # 通过线性层输出
            last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  # 计算交叉熵损失
        else:
            logits = self.output(h)  # 没有目标时，只进行推理
            last_loss = None

        return logits, h, last_loss  # 返回logits、最后的隐藏状态和损失


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 配置优化器，使用AdamW优化器
        param_dict = {pn: p for pn, p in self.named_parameters()}  # 获取所有模型参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}  # 过滤掉不需要梯度的参数
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # 筛选出需要weight decay的参数
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # 筛选出不需要weight decay的参数

        # 创建优化器组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)  # 统计需要weight decay的参数数量
        num_nodecay_params = sum(p.numel() for p in nodecay_params)  # 统计不需要weight decay的参数数量
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 创建AdamW优化器，选择是否使用fused优化器
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer  # 返回优化器

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估算模型的FLOP使用情况（MFU）"""
        # 估算每次迭代的FLOP数
        N = sum(p.numel() for p in self.parameters())  # 计算所有参数的总数
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T  # 每个token的FLOP数，参考PaLM论文附录B
        flops_per_fwdbwd = flops_per_token * T  # 每次前向和反向传播的FLOP数
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  # 每次迭代的FLOP数
        flops_achieved = flops_per_iter * (1.0/dt)  # 每秒的FLOP数
        flops_promised = 312e12  # A100 GPU的bfloat16峰值FLOP数（312 TFLOPS）
        mfu = flops_achieved / flops_promised  # 计算FLOP使用率
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        使用给定的prompt序列生成新序列，通过递归地将每次生成的token作为输入传入模型。
        """
        for _ in range(max_new_tokens):
            # 如果上下文序列太长，裁剪它使其不超过最大长度
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond)[0]  # 前向推理，获得logits
            logits = logits[:, -1, :]  # 只保留最后一步的logits
            if temperature == 0.0:
                # 选择概率最大的token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 使用给定的温度调整logits，控制输出的随机性
                logits = logits / temperature
                # 可选地裁剪logits，保留top_k的选项
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # 对logits进行softmax，转换为概率分布
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # 按照概率分布随机选择一个token
            # 将生成的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
        return idx  # 返回生成的序列

    @torch.inference_mode()
    def generate_withcache(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_k: int = 0,
        top_p: float = 0.9,
        eos_id: int = None,
        echo: bool = False,
        device: str = "cpu",
    ):
        """
        使用缓存进行文本生成。缓存用于存储之前计算的key和value，避免重复计算。
        """
        assert self.params.cache, "cache must be enabled for this method"  # 确保启用了缓存
        params = self.params
        bsz = len(prompt_tokens)  # 批次大小
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)  # 确保批次大小不超过最大值

        # 计算生成文本的总长度
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        assert min_prompt_len < total_len, "prompt is too long, not enough tokens left for generation!"

        # 初始化生成的tokens，填充为pad_id
        pad_id = -1
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)  # 填充前缀

        prev_pos = 0  # 上一个位置
        eos_reached = torch.tensor([False] * bsz, device=device)  # 判断是否已到达eos token
        input_text_mask = tokens != pad_id  # 创建掩码，标记有效的位置

        for cur_pos in range(min_prompt_len, total_len):
            # 获取当前序列的logits
            logits = self(tokens[:, prev_pos:cur_pos], start_pos=prev_pos, cache=True)[0]
            if temperature > 0:
                # 根据温度和top_k、top_p参数生成下一个token
                next_token = sample_top_k_top_p(logits[:, -1], top_k=top_k, top_p=top_p, temperature=temperature)
            else:
                # 选择最有可能的token
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # 只有在prompt已经生成的情况下才替换token
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token  # 更新生成的token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == eos_id
            )  # 判断是否达到eos
            prev_pos = cur_pos  # 更新位置
            if all(eos_reached):  # 如果所有样本都已生成eos token，则停止生成
                break
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            start = len(prompt_tokens[i])
            # 如果到达eos token，则截断生成的tokens
            if eos_id in toks:
                eos_idx = toks.index(eos_id)
                toks = toks[:eos_idx + 1]
            if echo:
                toks = prompt_tokens[i] + toks  # 如果echo为True，则将prompt_tokens加入生成结果
            out_tokens.append(toks)
        return out_tokens


def sample_top_k_top_p(logits, top_k=0, top_p=0.0, temperature=1.0):
    '''
    温度（temperature）：温度是用来调整logits的尺度，低温度值会使模型趋向于选择最可能的token，而高温度值则会使采样更具随机性。
    Top-k采样：从logits中选择前top_k个最可能的token，其它token的logits会被设置为负无穷大，确保它们不会被选中。
    Top-p (Nucleus)采样：选择一个最小的token集合，使得这些token的累积概率超过给定的阈值top_p，然后从这些token中进行采样。
    '''
    logits = logits / temperature  # 调整logits，使得它们符合给定的温度。温度控制采样的随机性，温度越低，模型的选择越集中在概率高的token上。

    if top_k > 0:
        # 可选地裁剪logits，保留前k个最有可能的选项
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 获取前top_k个最大值
        logits[logits < v[:, [-1]]] = -float('Inf')  # 将不在前top_k范围内的logits值设置为负无穷大，确保这些token不会被选择
        # 对logits进行softmax，转换为（归一化）概率分布
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # 从概率分布中按概率随机选择下一个token
    if 0 <= top_p <= 1.0:
        # Top-p (nucleus) 采样：从概率分布中选择最小的一组token，使得它们的累积概率大于top_p阈值
        probs = F.softmax(logits, dim=-1)  # 计算softmax概率
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # 对概率进行排序
        probs_sum = torch.cumsum(probs_sort, dim=-1)  # 计算排序后的累积概率
        mask = probs_sum - probs_sort > top_p  # 找到累积概率大于top_p的阈值位置
        probs_sort[mask] = 0.0  # 将超过top_p的部分概率设置为0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # 对剩余的概率进行归一化
        next_token = torch.multinomial(probs_sort, num_samples=1)  # 从修改后的概率分布中采样
        next_token = torch.gather(probs_idx, -1, next_token)  # 获取最终的token

    return next_token  # 返回采样的token


class TransformerWithValueHead(Transformer):
    '''
    value_head：这是一个线性层，它接收Transformer的输出（h）并将其映射到一个标量值。这通常用于回归任务，其中模型需要预测一个实数值（例如评分、估计等）。
    在forward方法中，首先调用父类的forward方法获取logits（用于分类或生成的输出）和隐藏状态h，然后将隐藏状态h传递给value_head来预测标量值。
    '''
    """ Transformer with an extra head for predicting a scalar value """

    def __init__(self, params: ModelArgs):
        super().__init__(params)
        self.value_head = nn.Linear(params.dim, 1)  # 添加一个线性层，用来预测标量值
        self._init_weights(self.value_head)  # 初始化value_head的权重

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, start_pos: int = 0,
                cache: bool = False) -> torch.Tensor:
        logits, h, last_loss = super().forward(tokens, targets, start_pos, cache)  # 调用父类的forward方法
        values = self.value_head(h).squeeze(-1)  # 使用value_head预测标量值（如回归任务中的目标值）
        return logits, h, values, last_loss  # 返回logits, hidden states, 预测的标量值和损失值
