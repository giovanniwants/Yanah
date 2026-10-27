# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Parity test for mlx_backend against stock mlx-lm, on a tiny random
Qwen3-Next (3 Gated DeltaNet layers + 1 full-attention layer, 4-bit
experts, 8-bit routers) plus a random MTP head.

  1. Streamed pool-path forward == mlx-lm forward (prompt logits).
  2. Streamed greedy decode == mlx-lm greedy decode, with a pool far smaller
     than the expert set (evictions, forecasts, adoption all active).
  3. Prefill sweep path: close to mlx-lm (grouped summation order differs).
  4. MTP speculation (depth 3, random head, so many rejections) == greedy,
     and the Gated DeltaNet + KV state after rollback == plain decode's.
  5. Per-token GDN stepping == fused update.

Runs on CPU (mlx[cpu]) or Apple Silicon.
    python -m moe_prefetch.test_mlx_backend
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_next import Model, ModelArgs
from mlx_lm.utils import load_model

from moe_prefetch.mlx_backend import MTPHead, build_runtime
from moe_prefetch.repack import repack
from moe_prefetch.speculative import MTPSpeculator

CFG = dict(model_type="qwen3_next", hidden_size=128, num_hidden_layers=4, intermediate_size=128,
           num_attention_heads=4, head_dim=32, num_key_value_heads=2, linear_num_value_heads=4,
           linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
           linear_conv_kernel_dim=4, num_experts=16, num_experts_per_tok=4, decoder_sparse_step=1,
           shared_expert_intermediate_size=64, mlp_only_layers=[], moe_intermediate_size=64,
           rms_norm_eps=1e-6, vocab_size=256, rope_theta=10000.0, partial_rotary_factor=0.25,
           max_position_embeddings=4096, norm_topk_prob=True, tie_word_embeddings=False,
           full_attention_interval=4)


def quantize(module):
    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
            return {"group_size": 64, "bits": 8}
        return True
    nn.quantize(module, group_size=64, bits=4, class_predicate=pred)


def make_checkpoint(d: str) -> None:
    mx.random.seed(0)
    args = ModelArgs.from_dict(CFG)
    model = Model(args)
    quantize(model)
    mtp = MTPHead(args)
    quantize(mtp)
    w = dict(tree_flatten(model.parameters()))
    w.update({f"mtp.{k}": v for k, v in tree_flatten(mtp.parameters())})
    mx.save_safetensors(os.path.join(d, "model.safetensors"), w)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({**CFG, "quantization": {"group_size": 64, "bits": 4}}, f)


def ref_greedy(model_dir, prompt, n):
    # strict=False: stock mlx-lm has no MTP module, so it ignores mtp.* keys
    model, _ = load_model(Path(model_dir), strict=False)
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    first = np.array(logits[0].astype(mx.float32))
    toks = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        logits = model(mx.array([[toks[-1]]]), cache=cache)
        toks.append(int(mx.argmax(logits[0, -1]).item()))
    return first, toks


def state_of(b):
    parts = []
    for i, c in enumerate(b.cache):
        if b.model.layers[i].is_linear:
            parts += [np.array(c[0].astype(mx.float32)).ravel(), np.array(c[1]).ravel()]
        else:
            k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
            parts += [np.array(k.astype(mx.float32)).ravel(), np.array(v.astype(mx.float32)).ravel()]
    return np.concatenate(parts)


def main():
    d = tempfile.mkdtemp()
    model_dir, exp_dir = os.path.join(d, "model"), os.path.join(d, "experts")
    os.makedirs(model_dir)
    make_checkpoint(model_dir)
    spec = repack(model_dir, exp_dir)
    print(f"repacked: record {spec.record_bytes} B x {spec.n_experts} experts x "
          f"{spec.n_layers}+{spec.n_mtp_layers} layers")
    rng = np.random.default_rng(3)
    prompt = [int(t) for t in rng.integers(0, 256, 12)]
    N = 24
    ref_first, ref_toks = ref_greedy(model_dir, prompt, N)

    # 1 + 2: pool path, tiny pool (40 slots of 80) so eviction/forecasting run
    orch, close = build_runtime(model_dir, exp_dir, slots=40, sweep_min_tokens=0,
                                spec_lanes=2, spec_slot_cap=8, mtp_norm_offset=0.0)
    logits, _ = orch.forward(prompt)
    err = np.abs(logits - ref_first).max()
    assert err < 1e-4, err
    print(f"[1] prompt logits match mlx-lm (max |diff| = {err:.2e})")
    orch.b.reset()
    toks = MTPSpeculator(orch, enabled=False, prefill_chunk=64).generate(prompt, N)
    assert toks == ref_toks, (toks, ref_toks)
    ps = orch.pool.stats
    print(f"[2] greedy decode matches mlx-lm over {N} tokens | pool hit {ps.hit_rate:.0%}, "
          f"evictions {ps.evictions}, forecasts adopted {ps.adopted}, discarded {ps.spec_discarded} "
          f"| lazy={orch.lazy} overlap={orch.overlap}")
    plain_toks = toks
    close()
    # Same check in every expert-execution mode, decode logits compared too.
    # 24 slots over 64 experts: every layer evicts, so a slot written while a
    # submitted graph still reads it would corrupt the result.
    ref_logits = None
    for mode in [dict(lazy_moe=False, overlap_resident=False), dict(lazy_moe=True, overlap_resident=False),
                 dict(lazy_moe=True, overlap_resident=True)]:
        orch, close = build_runtime(model_dir, exp_dir, slots=24, sweep_min_tokens=0, spec_lanes=2,
                                    spec_slot_cap=6, mtp_norm_offset=0.0, **mode)
        sp = MTPSpeculator(orch, enabled=False, prefill_chunk=64)
        toks = sp.generate(prompt, N)
        lg, _ = orch.forward([toks[-1]])
        close()
        assert toks == ref_toks, (mode, toks, ref_toks)
        if ref_logits is None:
            ref_logits = lg
        derr = np.abs(lg - ref_logits).max()
        assert derr == 0.0, (mode, derr)
        print(f"    {mode}: tokens == mlx-lm, next logits identical to synced mode "
              f"(evictions {orch.pool.stats.evictions})")

    # 3: sweep path for prompt passes >= 8 tokens
    orch, close = build_runtime(model_dir, exp_dir, slots=40, sweep_min_tokens=8,
                                spec_lanes=2, spec_slot_cap=8, mtp_norm_offset=0.0)
    logits, _ = orch.forward(prompt)
    err = np.abs(logits - ref_first).max()
    assert err < 1e-3, err
    print(f"[3] sweep prefill close to mlx-lm (max |diff| = {err:.2e}); "
          f"resident copies {orch.sweep.stats.resident_copies}, disk records {orch.sweep.stats.disk_records}, "
          f"admitted {orch.sweep.stats.admitted}")
    close()

    # 4: MTP speculation with rollback
    orch, close = build_runtime(model_dir, exp_dir, slots=40, sweep_min_tokens=0,
                                spec_lanes=2, spec_slot_cap=8, mtp_norm_offset=0.0)
    assert orch.b.has_mtp
    sp = MTPSpeculator(orch, max_depth=3, adaptive=False, prefill_chunk=64)
    toks = sp.generate(prompt, N)
    assert toks == plain_toks, (toks, plain_toks)
    consumed = len(prompt) + sp.stats.emitted          # positions fed to the state
    spec_state = state_of(orch.b)
    close()
    print(f"[4] MTP depth 3 output == greedy | {sp.stats.summary()}")

    orch, close = build_runtime(model_dir, exp_dir, slots=40, sweep_min_tokens=0,
                                spec_lanes=2, spec_slot_cap=8, mtp_norm_offset=0.0)
    MTPSpeculator(orch, enabled=False, prefill_chunk=64).generate(prompt, consumed - len(prompt) + 1)
    plain_state = state_of(orch.b)
    assert spec_state.shape == plain_state.shape
    serr = np.abs(spec_state - plain_state).max()
    assert serr < 1e-4, serr
    print(f"    state after rollback == plain decode at {consumed} positions (max |diff| = {serr:.2e})")

    # 6: batched decode == each sequence decoded alone. Different prompt
    # lengths (left-padded attention caches), staggered lengths (finished
    # rows are filtered out mid-run), tiny pool (every layer evicts).
    rng6 = np.random.default_rng(11)
    prompts = [[int(t) for t in rng6.integers(0, 256, L)] for L in (12, 5, 9)]
    limits = [20, 8, 14]
    alone = []
    for pr, lim in zip(prompts, limits):
        o, c = build_runtime(model_dir, exp_dir, slots=40, sweep_min_tokens=0, spec_lanes=2,
                             spec_slot_cap=8, mtp_norm_offset=0.0)
        alone.append(MTPSpeculator(o, enabled=False, prefill_chunk=64).generate(pr, lim))
        c()
    o, c = build_runtime(model_dir, exp_dir, slots=40, sweep_min_tokens=0, spec_lanes=2,
                         spec_slot_cap=8, mtp_norm_offset=0.0)
    steps = []
    batched = o.generate_batch(prompts, limits, prefill_chunk=64, step_log=steps)
    c()
    for i, (a_, b_) in enumerate(zip(alone, batched)):
        assert a_ == b_, (i, a_, b_)
    rows = [r for r, _ in steps]
    print(f"[6] batch of {len(prompts)} == each sequence decoded alone (rows per step {rows[0]}->{rows[-1]}, "
          f"{len(steps)} steps, filter exercised)")

    # 5
    dmax = max(orch.b.check_stepping(l) for l in range(3))
    assert dmax < 1e-5, dmax
    print(f"[5] per-token GDN stepping == fused update (max |diff| = {dmax:.2e})")
    close()
    print("\nall MLX backend checks passed")


if __name__ == "__main__":
    main()
