# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Expert geometry + on-disk layout.

The single biggest IO win is NOT the prefetcher, it's the file layout:
stock safetensors store each expert as 3 matrices (gate/up/down), each
with separate weight/scales/biases tensors => up to 9 scattered reads per
expert. Repack once so every expert is ONE contiguous, 16 KiB-aligned blob
(gate|up|down, with their quant scales), ordered by (layer, expert).
Then one expert == one pread, and experts that are often co-activated can
later be reordered to sit next to each other so reads coalesce.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

PAGE = 16 * 1024  # Apple Silicon VM page size

ExpertKey = Tuple[int, int]  # (layer, expert_id)


def align_up(n: int, a: int = PAGE) -> int:
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class ModelGeometry:
    """Defaults match Qwen3-Next-80B-A3B (verify against your config.json)."""
    n_layers: int = 48           # main decoder layers
    n_mtp_layers: int = 1        # MTP module is an extra MoE layer -> index n_layers
    n_experts: int = 512
    top_k: int = 10              # routed experts per token (+1 shared, kept resident)
    hidden: int = 2048
    moe_inter: int = 512
    bits: int = 4
    group_size: int = 64
    scale_bytes: int = 2         # fp16 scale + fp16 bias per group
    record_bytes: int = 0        # exact record size from the repack index (overrides the estimate)

    @property
    def total_moe_layers(self) -> int:
        return self.n_layers + self.n_mtp_layers

    @property
    def expert_param_count(self) -> int:
        return 3 * self.hidden * self.moe_inter  # gate, up, down

    @property
    def expert_nbytes(self) -> int:
        if self.record_bytes:
            return self.record_bytes
        p = self.expert_param_count
        packed = p * self.bits // 8
        quant_meta = (p // self.group_size) * self.scale_bytes * 2
        return packed + quant_meta

    @property
    def slot_nbytes(self) -> int:
        return align_up(self.expert_nbytes)

    def summary(self) -> str:
        mb = self.slot_nbytes / 2**20
        per_tok = mb * self.top_k * self.total_moe_layers
        total = mb * self.n_experts * self.total_moe_layers / 1024
        return (f"expert={mb:.2f} MiB, cold read/token={per_tok:.0f} MiB, "
                f"all experts={total:.1f} GiB")


@dataclass(frozen=True)
class ExpertLocation:
    file_idx: int
    offset: int
    nbytes: int


class ExpertLayout:
    """Maps (layer, expert) -> where its contiguous blob lives on disk."""

    def __init__(self, geometry: ModelGeometry, files: List[str],
                 locations: Dict[ExpertKey, ExpertLocation]):
        self.geometry = geometry
        self.files = files
        self.locations = locations

    def __getitem__(self, key: ExpertKey) -> ExpertLocation:
        return self.locations[key]

    def keys_for_layer(self, layer: int) -> Iterable[ExpertKey]:
        return ((layer, e) for e in range(self.geometry.n_experts))

    @classmethod
    def contiguous(cls, geometry: ModelGeometry, path: str) -> "ExpertLayout":
        """Layout produced by the repacker: one file, row-major (layer, expert)."""
        locs = {}
        stride = geometry.slot_nbytes
        for layer in range(geometry.total_moe_layers):
            for e in range(geometry.n_experts):
                off = (layer * geometry.n_experts + e) * stride
                locs[(layer, e)] = ExpertLocation(0, off, geometry.expert_nbytes)
        return cls(geometry, [path], locs)

    # Index file lets a repacker emit any order (e.g. co-activation clustered).
    def save_index(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "geometry": asdict(self.geometry),
                "files": self.files,
                "experts": [[l, e, loc.file_idx, loc.offset, loc.nbytes]
                            for (l, e), loc in sorted(self.locations.items())],
            }, f)

    @classmethod
    def load_index(cls, path: str) -> "ExpertLayout":
        with open(path) as f:
            d = json.load(f)
        locs = {(l, e): ExpertLocation(fi, off, n) for l, e, fi, off, n in d["experts"]}
        return cls(ModelGeometry(**d["geometry"]), d["files"], locs)
