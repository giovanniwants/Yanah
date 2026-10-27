"""
quantize_2bit.py checks on the tiny Qwen3-Next:
  1. the streamed engine on the 2-bit experts.bin == stock mlx-lm with its
     experts requantized the same way (prompt logits and greedy decode);
  2. an interrupted + resumed conversion writes the identical file;
  3. spec, size, and the reported requantization error.
    python test_quantize_2bit.py
"""
import filecmp
import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

import moe_prefetch.test_mlx_backend as tmb
from moe_prefetch.mlx_backend import build_runtime
from moe_prefetch.speculative import MTPSpeculator
from mlx_lm.utils import load_model
from quantize_2bit import convert


def stock_2bit(model_dir, bits=2, group=32):
    model, _ = load_model(Path(model_dir), strict=False)
    for L in model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(L.mlp.switch_mlp, name)
            deq = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=lin.group_size, bits=lin.bits)
            lin.weight, lin.scales, lin.biases = mx.quantize(deq, group_size=group, bits=bits)
            lin.group_size, lin.bits = group, bits
    return model


def main():
    d = tempfile.mkdtemp()
    md = os.path.join(d, "model"); os.makedirs(md)
    tmb.make_checkpoint(md)
    full, resumed = os.path.join(d, "full"), os.path.join(d, "resumed")
    fds0 = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
    spec = convert(md, full, min_free_gb=0, chunk=4)               # many reads per layer
    if fds0 is not None:
        leaked = len(os.listdir("/proc/self/fd")) - fds0
        assert leaked <= 0, f"{leaked} file descriptors leaked"
    convert(md, resumed, min_free_gb=0, max_layers=2)              # interrupted after 2 layers
    assert os.path.exists(os.path.join(resumed, "experts.bin.partial"))
    convert(md, resumed, min_free_gb=0)                            # resume
    assert filecmp.cmp(os.path.join(full, "experts.bin"), os.path.join(resumed, "experts.bin"), shallow=False)
    meta = json.load(open(os.path.join(full, "experts.json")))
    L = spec.n_layers + spec.n_mtp_layers
    assert meta["bits"] == 2 and meta["group_size"] == 32
    assert os.path.getsize(os.path.join(full, "experts.bin")) == L * spec.n_experts * spec.slot_bytes
    print(f"[1] 2-bit g32 record {spec.record_bytes} B (4-bit was 15360 B); resumed file identical; "
          f"rel error per layer {np.round(meta['rel_rms_error_vs_source'], 3).tolist()}")

    rng = np.random.default_rng(3)
    prompt = [int(t) for t in rng.integers(0, 256, 12)]
    ref = stock_2bit(md)
    ref_logits = np.array(ref(mx.array([prompt]))[0, -1].astype(mx.float32))
    cache = ref.make_cache()
    logits = ref(mx.array([prompt]), cache=cache)[0, -1]
    ref_toks = []
    for _ in range(16):
        t = int(mx.argmax(logits).item()); ref_toks.append(t)
        if len(ref_toks) == 16: break
        logits = ref(mx.array([[t]]), cache=cache)[0, -1]

    orch, close = build_runtime(md, full, slots=40, sweep_min_tokens=0, spec_lanes=2, spec_slot_cap=8,
                                mtp_norm_offset=0.0)
    assert orch.b.bits == 2 and orch.b.group_size == 32, (orch.b.bits, orch.b.group_size)
    lg, _ = orch.forward(prompt)
    err = np.abs(lg[-1] - ref_logits).max()
    assert err < 1e-4, err
    orch.b.reset()
    toks = MTPSpeculator(orch, enabled=False, prefill_chunk=64).generate(prompt, 16)
    close()
    assert toks == ref_toks, (toks, ref_toks)
    print(f"[2] streamed engine on the 2-bit experts.bin == stock mlx-lm with 2-bit experts "
          f"(logits max |diff| {err:.1e}, 16 greedy tokens identical)")
    # 3. the same pipeline at 3-bit g64 (half 2-bit's weight error, 17% more bytes)
    three = os.path.join(d, "three")
    convert(md, three, bits=3, group=64, min_free_gb=0)
    ref3 = stock_2bit(md, 3, 64)
    r3 = np.array(ref3(mx.array([prompt]))[0, -1].astype(mx.float32))
    orch, close = build_runtime(md, three, slots=40, sweep_min_tokens=0, spec_lanes=2, spec_slot_cap=8,
                                mtp_norm_offset=0.0)
    assert orch.b.bits == 3 and orch.b.group_size == 64
    lg3, _ = orch.forward(prompt)
    close()
    assert np.abs(lg3[-1] - r3).max() < 1e-4
    print(f"[3] --bits 3 --group-size 64 also matches stock mlx-lm exactly")
    print("\nall quantize_2bit checks passed")


if __name__ == "__main__":
    main()
