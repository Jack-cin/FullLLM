"""
Sample from the trained model with PyTorch
（用训练好的 checkpoint 做文本生成）

🛠️ 原版 sample.py 有两个致命问题，直接跑必崩：
  1. `exec(open('configurator.py').read())`：仓库里根本没有 configurator.py
     → FileNotFoundError（这是从 nanoGPT 抄过来的残留，nanoGPT 里才有这个文件）。
  2. `checkpoint_dict["config"]`：train.py 存的键是 "model_args"，取 "config" 会 KeyError。
现在改成标准 argparse 命令行，并同时兼容 "model_args" / "config" 两种 checkpoint 格式。

用法示例：
  python sample.py --checkpoint=out/ckpt.pt --start="Once upon a time" --max_new_tokens=200
  python sample.py --checkpoint=data/ckpt.pt --start=FILE:prompt.txt --temperature=0.8 --top_k=50
"""
import argparse
import os
import time
from contextlib import nullcontext

import torch

from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from tinystories import get_tokenizer_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="从 checkpoint 采样生成故事")
    parser.add_argument("--checkpoint", type=str, default="out/ckpt.pt", help="checkpoint 路径")
    parser.add_argument("--start", type=str, default="", help="提示词；也支持 FILE:prompt.txt 从文件读取")
    parser.add_argument("--num_samples", type=int, default=1, help="采样条数")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="每条最多生成多少 token")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度：<1 更确定，>1 更随机")
    parser.add_argument("--top_k", type=int, default=0, help="top-k 采样（0=关闭）")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p / nucleus 采样（<=0 关闭）")
    parser.add_argument("--tokenizer", type=str, default="", help="手动指定 tokenizer.model 路径")
    parser.add_argument("--seed", type=int, default=1337, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda / cpu / mps")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"], help="推理精度")
    parser.add_argument("--compile", action="store_true", default=False, help="用 torch.compile 加速")
    return parser.parse_args()


def load_state_dict_compat(model, state_dict):
    """
    🛠️ 兼容 torch.compile 存出来的 `_orig_mod.` 前缀：train.py 里用 --compile 训练时，
    ckpt 中的参数名会带这个前缀，不剥掉的话 strict=False 会把所有权重都当成“缺失”，
    生成的就是一个随机初始化的模型（却完全不报错）。
    """
    unwanted_prefix = "_orig_mod."
    if any(k.startswith(unwanted_prefix) for k in state_dict):
        state_dict = {k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k: v
                      for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in info.missing_keys if "freqs" not in k]
    if len(missing) > 0.5 * len(list(model.state_dict())):
        raise RuntimeError(f"checkpoint 与模型结构不匹配，缺失 {len(missing)} 项参数：{missing[:5]}")
    return info


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    assert os.path.isfile(args.checkpoint), f"checkpoint 不存在：{args.checkpoint}"
    checkpoint_dict = torch.load(args.checkpoint, map_location="cpu")

    # 🛠️ train.py 存的是 "model_args"，早期 llama2.c 风格的 ckpt 存的是 "config"，两者都支持
    saved_args = checkpoint_dict.get("model_args") or checkpoint_dict.get("config") or {}
    model_kwargs = dict(saved_args) if isinstance(saved_args, dict) else vars(saved_args)
    # 只保留 ModelArgs 认识的字段，避免 ckpt 里的训练超参（batch_size 等）把构造搞崩
    valid = set(ModelArgs().__dict__.keys())
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid}
    model_kwargs["cache"] = True  # 推理必须开 KV cache，否则每步重算整段前缀
    model_kwargs["dropout"] = 0.0
    gptconf = ModelArgs(**model_kwargs)

    model = Transformer(gptconf)
    info = load_state_dict_compat(model, checkpoint_dict["model"])
    print(f"✓ 已加载 {args.checkpoint}（缺失 {len(info.missing_keys)} / 多余 {len(info.unexpected_keys)} 项）")
    model.eval()
    model.to(args.device)
    if args.compile:
        print("Compiling the model...")
        model = torch.compile(model)

    # 定位分词器模型
    if args.tokenizer:
        tokenizer_model = args.tokenizer
    else:
        vocab_source = model_kwargs.get("vocab_source", "custom")
        query_vocab_size = 0 if vocab_source == "llama2" else gptconf.vocab_size
        tokenizer_model = get_tokenizer_model_path(query_vocab_size)
    enc = Tokenizer(tokenizer_model=tokenizer_model)

    start = args.start
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = enc.encode(start, bos=True, eos=False)

    print(f"prompt: {start!r} | prompt tokens: {len(start_ids)} | device: {args.device}")
    t0 = time.time()
    generated_tokens = 0
    with torch.no_grad(), ctx:
        for k in range(args.num_samples):
            y = model.generate_withcache(
                [start_ids],
                max_gen_len=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p if args.top_p > 0 else 0.0,
                eos_id=enc.bos_id,
                echo=False,
                device=args.device,
            )
            generated_tokens += len(y[0])
            print(enc.decode(y[0]).strip())
            print("---------------")
    dt = time.time() - t0
    print(f"Generation time cost: {dt:.02f}s | {generated_tokens / max(dt, 1e-9):.1f} tok/s")


if __name__ == "__main__":
    main()
