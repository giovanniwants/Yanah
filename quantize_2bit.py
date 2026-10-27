"""
Requantize Qwen3-Next's routed experts to 2-bit for the streamed engine.

Scheme (as AliceAI-2bit): routed experts affine 2-bit, group size 32;
everything else (attention, Gated DeltaNet, shared experts, embeddings,
lm_head) stays 4-bit g64; routers stay 8-bit. Only the experts change, so
the output is just a new experts.bin + experts.json. The engine keeps
loading the dense weights from the 4-bit checkpoint:

    python quantize_2bit.py                     # paths from moe_prefetch/paths.py
    python run_mlx_test.py --experts ~/Desktop/yanah_models/Qwen3-80B-2bit-experts
    python mlx_server.py   --experts ~/Desktop/yanah_models/Qwen3-80B-2bit-experts

Quality note: this dequantizes the 4-bit g64 experts and quantizes them
again (as AliceAI's convert_f80b_mixed.py does). AliceAI's published model
was instead built once from the original BF16 weights, to avoid quantizing
twice. That needs the ~160 GB BF16 Qwen3-Next checkpoint. Measure quality
before switching (run_mlx_test.py teacher-forced comparison).

Size: a 2-bit g32 expert is 1.125 MiB vs 1.69 MiB at 4-bit g64 (g32's
scales and biases are a third of it), so 0.67x per read and 1.5x the
experts per GB of pool. The whole file is ~27 GiB.

Resumable: progress goes to experts.bin.partial in whole layers; rerun
the same command to continue. Reads the stacked safetensors tensors by
byte range, a chunk of experts at a time, so it never holds a whole layer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from moe_prefetch.layout import PAGE, align_up
from moe_prefetch.paths import EXPERTS_2BIT, MODEL_4BIT
from moe_prefetch.repack import PIECES, PROJS, Piece, RecordSpec, _prefix, read_headers

_NP = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def _progress(total: int):
    try:
        from tqdm import tqdm
        return tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, smoothing=0.05,
                    desc="2-bit experts")
    except ImportError:                                   # minimal fallback bar
        class Bar:
            def __init__(self): self.n, self.t0 = 0, time.time()
            def update(self, k):
                self.n += k
                f = self.n / total
                rate = self.n / max(1e-9, time.time() - self.t0)
                eta = (total - self.n) / max(1, rate)
                sys.stderr.write(f"\r[{'#' * int(30 * f):<30}] {f:6.1%} {rate / 2**20:6.0f} MiB/s ETA {eta / 60:5.1f} min")
            def set_postfix_str(self, s): pass
            def close(self): sys.stderr.write("\n")
        return Bar()


class Source:
    """Byte-range reader for one layer's stacked expert tensor."""

    def __init__(self, model_dir: str):
        self.hdrs = read_headers(model_dir)
        self.fds = {}

    def read(self, name: str, e0: int, e1: int, n_experts: int):
        import mlx.core as mx
        f, ent, base = self.hdrs[name]
        b, e = ent["data_offsets"]
        per = (e - b) // n_experts
        fd = self.fds.get(f)
        if fd is None:                      # open each shard once (setdefault would open on every call)
            fd = self.fds[f] = os.open(f, os.O_RDONLY)
        raw = os.pread(fd, per * (e1 - e0), base + b + per * e0)
        a = np.frombuffer(raw, dtype=_NP[ent["dtype"]]).reshape(e1 - e0, *ent["shape"][1:])
        arr = mx.array(a)
        return arr.view(mx.bfloat16) if ent["dtype"] == "BF16" else arr

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def out_spec(src: RecordSpec, src_bits: int, bits: int, group: int) -> RecordSpec:
    pieces = []
    by = {p.name: p for p in src.pieces}
    for proj in PROJS:
        w = by[f"{proj}.weight"]
        out_dim, in_dim = w.shape[0], w.shape[1] * 32 // src_bits
        if in_dim % group or (in_dim * bits) % 32:
            raise ValueError(f"{proj}: input dim {in_dim} is not divisible by group size {group}")
        sdt = by[f"{proj}.scales"].dtype
        item = np.dtype(_NP[sdt]).itemsize
        pieces += [Piece(f"{proj}.weight", "U32", [out_dim, in_dim * bits // 32], out_dim * in_dim * bits // 8),
                   Piece(f"{proj}.scales", sdt, [out_dim, in_dim // group], out_dim * (in_dim // group) * item),
                   Piece(f"{proj}.biases", sdt, [out_dim, in_dim // group], out_dim * (in_dim // group) * item)]
    return RecordSpec(pieces, group, bits, src.n_layers, src.n_mtp_layers, src.n_experts)


def convert(model_dir: str, out_dir: str, bits: int = 2, group: int = 32, chunk: int = 64,
            min_free_gb: float = 10.0, force: bool = False, max_layers: int = None) -> RecordSpec:
    import mlx.core as mx
    from moe_prefetch.repack import build_spec
    src, _ = build_spec(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    q = cfg.get("quantization", {})
    sbits, sgroup = q.get("bits", 4), q.get("group_size", 64)
    spec = out_spec(src, sbits, bits, group)
    L, E = spec.n_layers + spec.n_mtp_layers, spec.n_experts
    layer_bytes = E * spec.slot_bytes
    total = L * layer_bytes
    os.makedirs(out_dir, exist_ok=True)
    final, part = os.path.join(out_dir, "experts.bin"), os.path.join(out_dir, "experts.bin.partial")
    meta_part = os.path.join(out_dir, "experts.partial.json")
    done = 0
    if os.path.exists(part):
        ok = os.path.exists(meta_part) and json.load(open(meta_part)) == spec.to_json()
        size = os.path.getsize(part)
        if ok:
            done = size // layer_bytes
            print(f"resuming: {done}/{L} layers already written", file=sys.stderr)
        else:
            raise SystemExit(f"{part} is from a different spec; delete it to start over")
    free = shutil.disk_usage(out_dir).free
    need = total - done * layer_bytes
    if not force and free - need < min_free_gb * 2**30:
        raise SystemExit(f"need {need / 2**30:.1f} GiB, disk has {free / 2**30:.1f} GiB free; that would leave "
                         f"under {min_free_gb:.0f} GiB for macOS. Free space first, or pass --force / --min-free-gb")
    json.dump(spec.to_json(), open(meta_part, "w"))
    print(f"source: {model_dir} experts {sbits}-bit g{sgroup}, record {src.record_bytes / 2**20:.3f} MiB\n"
          f"output: {out_dir} experts {bits}-bit g{group}, record {spec.record_bytes / 2**20:.3f} MiB "
          f"({spec.record_bytes / src.record_bytes:.2f}x), {L} layers x {E} experts = {total / 2**30:.1f} GiB",
          file=sys.stderr)
    s = Source(model_dir)
    bar = _progress(total)
    bar.update(done * layer_bytes)
    pad = b"\0" * (spec.slot_bytes - spec.record_bytes)
    errs = []
    if os.path.exists(os.path.join(out_dir, "experts.errors.json")) and done:
        errs = json.load(open(os.path.join(out_dir, "experts.errors.json")))[:done]
    with open(part, "r+b" if done else "wb") as out:
        out.truncate(done * layer_bytes)
        out.seek(done * layer_bytes)
        for layer in range(done, L if max_layers is None else min(L, max_layers)):
            pre = _prefix(layer, spec.n_layers)
            num = den = 0.0
            for e0 in range(0, E, chunk):
                e1 = min(E, e0 + chunk)
                parts = []
                for proj in PROJS:
                    w, sc, bi = (s.read(f"{pre}.{proj}.{k}", e0, e1, E) for k in ("weight", "scales", "biases"))
                    deq = mx.dequantize(w, sc, bi, group_size=sgroup, bits=sbits)
                    qw, qs, qb = mx.quantize(deq, group_size=group, bits=bits)
                    back = mx.dequantize(qw, qs, qb, group_size=group, bits=bits)
                    d = (back.astype(mx.float32) - deq.astype(mx.float32))
                    num_, den_ = mx.sum(d * d), mx.sum(deq.astype(mx.float32) ** 2)
                    mx.eval(qw, qs, qb, num_, den_)
                    num += num_.item(); den += den_.item()
                    parts.append([np.array(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x) for x in (qw, qs, qb)])
                buf = bytearray()
                for i in range(e1 - e0):
                    for proj_parts in parts:
                        for arr in proj_parts:
                            buf += arr[i].tobytes()
                    buf += pad
                out.write(buf)
                bar.update(len(buf))
            out.flush()
            os.fsync(out.fileno())
            errs.append(float(np.sqrt(num / max(den, 1e-30))))
            json.dump(errs, open(os.path.join(out_dir, "experts.errors.json"), "w"))
            bar.set_postfix_str(f"layer {layer + 1}/{L} rel err {errs[-1]:.3f}")
    bar.close()
    s.close()
    if len(errs) < L:
        print(f"stopped after {len(errs)}/{L} layers (rerun to resume)", file=sys.stderr)
        return spec
    os.replace(part, final)
    meta = spec.to_json()
    meta.update(source=os.path.abspath(model_dir),
                scheme=dict(experts=dict(bits=bits, group_size=group), source_experts=dict(bits=sbits, group_size=sgroup),
                            other="unchanged (read from the source checkpoint)"),
                rel_rms_error_vs_source=errs)
    json.dump(meta, open(os.path.join(out_dir, "experts.json"), "w"), indent=1)
    os.remove(meta_part)
    print(f"done: {final} ({os.path.getsize(final) / 2**30:.1f} GiB); requantization error vs the 4-bit experts "
          f"(relative RMS) median {np.median(errs):.3f}, max {max(errs):.3f}", file=sys.stderr)
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(MODEL_4BIT), help="MLX 4-bit checkpoint (safetensors)")
    ap.add_argument("--out", default=str(EXPERTS_2BIT))
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64, help="experts per GPU batch")
    ap.add_argument("--min-free-gb", type=float, default=10.0, help="disk space to leave free")
    ap.add_argument("--force", action="store_true", help="skip the free-space check")
    ap.add_argument("--max-layers", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    convert(os.path.expanduser(a.src), os.path.expanduser(a.out), a.bits, a.group_size, a.chunk,
            a.min_free_gb, a.force, a.max_layers)


if __name__ == "__main__":
    main()
