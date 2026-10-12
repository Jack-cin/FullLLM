"""
冒烟测试 / 回归测试：不依赖 GPU、不依赖 TinyStories 原始数据集，
用合成数据把「数据 → 模型 → 训练 → 存 ckpt → 续训 → 导出 → 采样 → PPO step」整条链路跑通。

运行方式：
    pytest tests/smoke_test.py -v
    python tests/smoke_test.py          # 直接运行（无 pytest 也能跑）

每个测试都对应着一次具体的修复，注释里写清了“原来会怎么坏”。
"""
import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model import (  # noqa: E402
    ModelArgs,
    Transformer,
    TransformerWithValueHead,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from ppo import PPOConfig, PPOTrainer  # noqa: E402
from tinystories import get_tokenizer_model_path  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402
from train import (  # noqa: E402
    TinyStoriesTrainer,
    TrainConfig,
    config_to_dict,
    load_model_state,
    resolve_ckpt_path,
)

TINY = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=512,
            multiple_of=4, max_seq_len=64, dropout=0.0)


def tiny_args(**over) -> ModelArgs:
    kw = dict(TINY)
    kw.update(over)
    return ModelArgs(**kw)


@contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """
    造一个“迷你 data 目录”：
        data/tok512.model      （软链到仓库里真实的分词器，512 词表）
        data/tok512/shard_*.bin（合成的 uint16 token 流，模拟 pretokenize 的产物）
    这样就不需要下载 1.5GB 的 TinyStories 原始数据。
    """
    root = tmp_path_factory.mktemp("fullllm_smoke")
    (root / "data" / "tok512").mkdir(parents=True)
    os.symlink(REPO / "data" / "tok512.model", root / "data" / "tok512.model")
    rng = np.random.default_rng(0)
    for i in range(2):  # 训练/验证各自需要一个 shard（train 用 [1:]，val 用 [:1]）
        arr = rng.integers(3, 512, size=64 * 60, dtype=np.uint16)
        arr.tofile(root / "data" / "tok512" / f"shard_{i:02d}.bin")
    return root


# ---------------------------------------------------------------------------
# 1. 纯工具函数：配置序列化 / ckpt 定位 / 权重前缀
# ---------------------------------------------------------------------------
def test_config_to_dict_handles_namespace():
    """原代码 asdict(arg) 会直接抛 TypeError，第一次保存 ckpt 就崩。"""
    arg = argparse.Namespace(dim=64, batch_size=8)
    assert config_to_dict(arg) == {"dim": 64, "batch_size": 8}
    with pytest.raises(TypeError):  # 确认这就是修复前的行为
        asdict(arg)


def test_resolve_ckpt_path(tmp_path):
    """PPO 阶段存的是 ppo_ckpt.pt，原代码写死找 ckpt.pt。"""
    (tmp_path / "ckpt.pt").write_bytes(b"")
    assert Path(resolve_ckpt_path(str(tmp_path), ppo=False)).name == "ckpt.pt"
    assert Path(resolve_ckpt_path(str(tmp_path), ppo=True)).name == "ckpt.pt"

    (tmp_path / "ppo_ckpt.pt").write_bytes(b"")
    assert Path(resolve_ckpt_path(str(tmp_path), ppo=True)).name == "ppo_ckpt.pt"
    explicit = tmp_path / "cko.pt"
    explicit.write_bytes(b"")
    assert resolve_ckpt_path(str(tmp_path), ppo=True, ckpt_path=str(explicit)) == str(explicit)


def test_load_model_state_strips_compile_prefix():
    """--compile 训练出的 ckpt 带 _orig_mod. 前缀，不剥离就静默丢掉全部权重。"""
    model = Transformer(tiny_args())
    state = {f"_orig_mod.{k}": v.clone() for k, v in model.state_dict().items()}
    for k, v in state.items():
        if v.is_floating_point:
            v.add_(1.0)  # 扰动，确保断言能区分“真的加载了”和“没加载”

    load_model_state(model, state, master_process=False)
    loaded = model.state_dict()
    for k, v in loaded.items():
        if v.is_floating_point:
            assert torch.allclose(v, state[f"_orig_mod.{k}"]), k


def test_load_model_state_raises_when_nothing_matches():
    """权重完全对不上时必须报错，而不是静默从头训。"""
    model = Transformer(tiny_args())
    bogus = {f"not_a_param.{i}": torch.zeros(3) for i in range(200)}
    with pytest.raises(RuntimeError):
        load_model_state(model, bogus, master_process=False)


# ---------------------------------------------------------------------------
# 2. 模型前向：RoPE / 因果 mask / KV cache 的等价性
# ---------------------------------------------------------------------------
def reference_rotary(x, freqs_cos, freqs_sin):
    """逐元素实现（三重循环），作为 RoPE 的参照。"""
    b, t, h, d = x.shape
    out = torch.empty_like(x)
    for bi in range(b):
        for ti in range(t):
            for hi in range(h):
                for i in range(d // 2):
                    r, im = x[bi, ti, hi, 2 * i], x[bi, ti, hi, 2 * i + 1]
                    c, s = freqs_cos[ti, i], freqs_sin[ti, i]
                    out[bi, ti, hi, 2 * i] = r * c - im * s
                    out[bi, ti, hi, 2 * i + 1] = r * s + im * c
    return out


def test_rotary_equivalence_fp32_and_bf16():
    head_dim, T = 16, 7
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, T)
    for dtype in (torch.float32, torch.bfloat16):
        xq = torch.randn(2, T, 3, head_dim, dtype=dtype)
        xk = torch.randn(2, T, 3, head_dim, dtype=dtype)
        q, k = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        assert q.dtype == dtype and k.dtype == dtype
        ref = reference_rotary(xq.float(), freqs_cos, freqs_sin)
        tol = 1e-5 if dtype == torch.float32 else 2e-2  # bf16 只有 8 位有效数字
        assert torch.allclose(q.float(), ref, atol=tol), (q - ref).abs().max()


def test_sdpa_is_causal_matches_explicit_mask():
    """优化点：用 is_causal=True 代替手工构造 [T,T] 的 -inf mask，结果必须完全一致。"""
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    fast = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
    slow = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(fast, slow, atol=1e-6)


def test_kv_cache_decode_matches_full_forward():
    """带 cache 的增量解码 == 一次性整段前向（验证 mask / RoPE 偏移 / 因果快路径都没改语义）。"""
    torch.manual_seed(0)
    model = Transformer(tiny_args(cache=True)).eval()
    tokens = torch.randint(3, 512, (2, 9))

    with torch.no_grad():
        full_logits, _, _ = model(tokens)
        stepped = []
        for t in range(tokens.shape[1]):
            step_logits, _, _ = model(tokens[:, t:t + 1], start_pos=t, cache=True)
            stepped.append(step_logits)
    stepped_logits = torch.cat(stepped, dim=1)
    assert stepped_logits.shape == full_logits.shape
    assert torch.allclose(stepped_logits, full_logits, atol=1e-5), \
        (stepped_logits - full_logits).abs().max()


def test_generate_withcache_matches_greedy_reference():
    """KV cache 生成的结果，应与“每步重算整段前缀”的贪心解码一致。"""
    torch.manual_seed(0)
    model = Transformer(tiny_args(cache=True)).eval()
    prompt = torch.randint(3, 512, (1, 4)).tolist()

    with torch.no_grad():
        cached = model.generate_withcache(deepcopy(prompt), max_gen_len=12, temperature=0.0,
                                          top_k=0, top_p=0.0, eos_id=0, echo=False, device="cpu")[0]
        # 参照实现：不走 cache，每步都把整段前缀喂进去
        idx = torch.tensor(prompt, dtype=torch.long)
        for _ in range(12):
            logits, _, _ = model(idx)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        reference = idx[0].tolist()

    # generate_withcache 返回的是「prompt + 新生成」的完整序列（echo 只影响是否再拼一次 prompt）
    assert cached[len(prompt[0]):] == reference[len(prompt[0]):]


# ---------------------------------------------------------------------------
# 3. 训练 / 续训 / 导出 全链路
# ---------------------------------------------------------------------------
def _train_args(**over):
    argv = [
        "train.py", "--out_dir", "out_smoke", "--device", "cpu", "--dtype", "float32",
        "--vocab_source", "custom", "--vocab_size", "512", "--dim", "64", "--n_layers", "2",
        "--n_heads", "4", "--n_kv_heads", "2", "--multiple_of", "4", "--max_seq_len", "32",
        "--batch_size", "4", "--gradient_accumulation_steps", "1", "--max_iters", "3",
        "--eval_interval", "1", "--eval_iters", "1", "--log_interval", "1",
        "--warmup_iters", "1", "--learning_rate", "1e-3", "--num_workers", "2",
        "--always_save_checkpoint",
    ]
    for k, v in over.items():
        if isinstance(v, bool):  # store_true 类开关不能写成 --flag=True，argparse 会报错
            if v:
                argv.append(f"--{k}")
        else:
            argv.append(f"--{k}={v}")
    with mock.patch.object(sys, "argv", argv):
        return TrainConfig()


def test_training_smoke_writes_checkpoint_and_export(workdir, capsys):
    """
    端到端：合成数据 + 迷你模型跑 4 步训练。
    覆盖修复：_build_dloader 命名错、asdict(Namespace) 崩溃、num_workers 可配置、model.bin 导出。
    """
    with chdir(workdir):
        arg = _train_args()
        trainer = TinyStoriesTrainer(arg)
        trainer.train()

        ckpt_path = workdir / "out_smoke" / "ckpt.pt"
        assert ckpt_path.exists(), "训练结束没有产出 ckpt.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert isinstance(ckpt["model_args"], dict)
        assert ckpt["model_args"]["dim"] == 64
        assert "optimizer" in ckpt and ckpt["iter_num"] >= 1
        assert (workdir / "out_smoke" / "model.bin").exists(), "没有导出 model.bin"

        out = capsys.readouterr().out
        assert "每次迭代处理 token 数" in out


def test_resume_from_compiled_checkpoint_restores_weights(workdir):
    """
    覆盖修复：续训时读取带 _orig_mod. 前缀的 ckpt（--compile 训练产物）。
    修复前：strict=False 静默丢弃全部权重，等于用随机权重继续训。
    """
    src = workdir / "out_smoke" / "ckpt.pt"
    assert src.exists(), "依赖 test_training_smoke_writes_checkpoint_and_export 先生成 ckpt"

    with chdir(workdir):
        ckpt = torch.load(src, map_location="cpu")
        # 模拟 torch.compile 的命名
        ckpt["model"] = {f"_orig_mod.{k}": v for k, v in ckpt["model"].items()}
        torch.save(ckpt, workdir / "out_smoke" / "ckpt_compiled.pt")

        arg = _train_args(init_from="resume", max_iters="0",
                          ckpt_path=str(workdir / "out_smoke" / "ckpt_compiled.pt"))
        trainer = TinyStoriesTrainer(arg)

    model_state = trainer.raw_model.state_dict()
    for k, v in ckpt["model"].items():
        plain = k[len("_orig_mod."):]
        if plain in model_state and model_state[plain].is_floating_point:
            assert torch.allclose(model_state[plain], v), plain


def test_resume_prefers_ppo_ckpt_for_ppo_stage(workdir):
    """PPO 阶段应优先找 ppo_ckpt.pt（原代码只会找 out_dir/ckpt.pt）。"""
    with chdir(workdir):
        ckpt = torch.load(workdir / "out_smoke" / "ckpt.pt", map_location="cpu")
        # PPO 模型多了 value_head，把它补进 state_dict 才能严格加载
        model = TransformerWithValueHead(tiny_args(cache=True))
        state = dict(ckpt["model"])
        state.update({k: v for k, v in model.state_dict().items()
                      if k not in state and v.is_floating_point})
        ckpt["model"] = state
        torch.save(ckpt, workdir / "out_smoke" / "ppo_ckpt.pt")

        arg = _train_args(init_from="resume", max_iters="0", ppo=True)
        trainer = TinyStoriesTrainer(arg)
        assert Path(trainer.ckpt_path).name == "ppo_ckpt.pt"
        assert isinstance(trainer.raw_model, TransformerWithValueHead)


# ---------------------------------------------------------------------------
# 4. PPO：step() 能真正跑起来（原代码 backward_batch_size=None 直接 TypeError）
# ---------------------------------------------------------------------------
def test_ppo_step_runs():
    torch.manual_seed(0)
    model = TransformerWithValueHead(tiny_args())
    ref_model = deepcopy(model)
    enc = Tokenizer(str(REPO / "data" / "tok512.model"))
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=1e-4,
                                           betas=(0.9, 0.95), device_type="cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    config = PPOConfig(batch_size=4, mini_batch_size=2, gradient_accumulation_steps=1,
                       ppo_epochs=2, vf_coef=0.1, max_grad_norm=1.0, whiten_rewards=True,
                       early_stopping=False, steps=1)
    trainer = PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=enc,
                         optimizer=optimizer, scaler=scaler, is_ddp=False)
    # 运行时批量大小必须被自动补上（修复点）
    assert trainer.config.backward_batch_size == 2
    assert trainer.config.global_batch_size == 4

    queries = [torch.randint(3, 512, (5,)) for _ in range(4)]
    responses = [torch.randint(3, 512, (7,)) for _ in range(4)]
    scores = [torch.randn(1) for _ in range(4)]
    stats = trainer.step(queries=queries, responses=responses, scores=scores)

    for key in ("ppo/loss/policy", "ppo/policy/entropy", "ppo/kl_ref", "ppo/reward_all"):
        assert key in stats, f"缺少统计量 {key}"
        assert np.isfinite(stats[key]), (key, stats[key])
    assert all(np.isfinite(v) for v in stats.values() if isinstance(v, float))


# ---------------------------------------------------------------------------
# 5. sample.py 命令行
# ---------------------------------------------------------------------------
def test_sample_cli_runs():
    """原 sample.py 依赖不存在的 configurator.py，且读 ckpt['config'] 会 KeyError。"""
    cmd = [sys.executable, str(REPO / "sample.py"),
           "--checkpoint", str(REPO / "data" / "ckpt.pt"), "--device", "cpu",
           "--max_new_tokens", "20", "--num_samples", "1", "--temperature", "0.8", "--top_k", "50"]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "tok/s" in proc.stdout
    assert "已加载" in proc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
