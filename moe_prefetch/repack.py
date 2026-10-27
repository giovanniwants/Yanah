# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Repack an MLX-converted Qwen3-Next checkpoint's experts into one record per
expert: the 9 pieces (gate/up/down x weight/scales/biases) back to back,
padded to 16 KiB, ordered (layer, expert). MTP experts, if the checkpoint has
them, become layer index n_layers.

Reads safetensors headers directly and copies byte ranges with pread, so it
never materialises a whole stacked tensor (works on an 8 GB Mac).

    python -m moe_prefetch.repack [MODEL_DIR OUT_DIR]   # defaults: paths.py (4-bit)

Writes out/experts.bin and out/experts.json (record spec + layout).
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from .layout import PAGE, align_up

PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
PIECES = [f"{p}.{q}" for p in PROJS for q in PARTS]   # on-disk order inside a record


@dataclass
class Piece:
    name: str
    dtype: str           # safetensors dtype: U32, BF16, F16, F32
    shape: List[int]     # per-expert shape
    nbytes: int


@dataclass
class RecordSpec:
    pieces: List[Piece]
    group_size: int
    bits: int
    n_layers: int        # main MoE layers
    n_mtp_layers: int
    n_experts: int

    @property
    def record_bytes(self) -> int:
        return sum(p.nbytes for p in self.pieces)

    @property
    def slot_bytes(self) -> int:
        return align_up(self.record_bytes, PAGE)

    def to_json(self) -> dict:
        return {**asdict(self), "record_bytes": self.record_bytes, "slot_bytes": self.slot_bytes}

    @classmethod
    def from_json(cls, d: dict) -> "RecordSpec":
        return cls([Piece(**p) for p in d["pieces"]], d["group_size"], d["bits"],
                   d["n_layers"], d["n_mtp_layers"], d["n_experts"])


def read_headers(model_dir: str) -> Dict[str, Tuple[str, dict, int]]:
    """tensor name -> (file, header entry, absolute data offset)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, ent in hdr.items():
            if name != "__metadata__":
                out[name] = (f, ent, 8 + n)
    return out


def _prefix(layer: int, n_layers: int) -> str:
    return (f"model.layers.{layer}.mlp.switch_mlp" if layer < n_layers
            else f"mtp.layers.{layer - n_layers}.mlp.switch_mlp")


def build_spec(model_dir: str) -> Tuple[RecordSpec, Dict]:
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    hdrs = read_headers(model_dir)
    n_layers = cfg["num_hidden_layers"]
    E = cfg["num_experts"]
    n_mtp = 1 if any(k.startswith("mtp.layers.0.mlp.switch_mlp.") for k in hdrs) else 0
    pieces = []
    for pc in PIECES:
        _, ent, _ = hdrs[f"{_prefix(0, n_layers)}.{pc}"]
        b, e = ent["data_offsets"]
        assert ent["shape"][0] == E and (e - b) % E == 0, pc
        pieces.append(Piece(pc, ent["dtype"], ent["shape"][1:], (e - b) // E))
    q = cfg.get("quantization", {})
    return RecordSpec(pieces, q.get("group_size", 64), q.get("bits", 4), n_layers, n_mtp, E), hdrs


def repack(model_dir: str, out_dir: str) -> RecordSpec:
    spec, hdrs = build_spec(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    fds: Dict[str, int] = {}
    pad = b"\0" * (spec.slot_bytes - spec.record_bytes)
    total = spec.n_layers + spec.n_mtp_layers
    with open(os.path.join(out_dir, "experts.bin"), "wb") as out:
        for layer in range(total):
            srcs = []
            for p in spec.pieces:
                f, ent, base = hdrs[f"{_prefix(layer, spec.n_layers)}.{p.name}"]
                if ent["dtype"] != p.dtype or ent["shape"][1:] != p.shape:
                    raise ValueError(f"layer {layer} {p.name}: shape/dtype differs from layer 0")
                if f not in fds:                # open each shard once
                    fds[f] = os.open(f, os.O_RDONLY)
                srcs.append((fds[f], base + ent["data_offsets"][0], p.nbytes))
            for e in range(spec.n_experts):
                for fd, off, nb in srcs:
                    out.write(os.pread(fd, nb, off + e * nb))
                out.write(pad)
            print(f"\rlayer {layer + 1}/{total}", end="", file=sys.stderr)
    print(file=sys.stderr)
    for fd in fds.values():
        os.close(fd)
    with open(os.path.join(out_dir, "experts.json"), "w") as f:
        json.dump(spec.to_json(), f, indent=1)
    return spec


if __name__ == "__main__":
    from .paths import EXPERTS_4BIT, MODEL_4BIT
    args = sys.argv[1:]
    repack(args[0] if args else str(MODEL_4BIT), args[1] if len(args) > 1 else str(EXPERTS_4BIT))
