# SPDX-License-Identifier: FSL-1.1-ALv2
"""
8 GB memory plan and decode-speed ceiling for Qwen3-Next-80B-A3B (4-bit).

Every constant here is an ESTIMATE until measured on the target Mac; the
point of this module is to make the budget explicit and to fail loudly
instead of swapping (SlotStream refuses to start below its floor for the
same reason). Replace the estimates with numbers from your own weights
(sum of non-expert tensor bytes), `vm_stat`, and a cold-read benchmark
(never-repeat offsets, F_NOCACHE: see SlotStream's Tools/coldread.c; a file
you just wrote is still in the page cache and will read at fake speeds).

Run:  python -m moe_prefetch.plan
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .layout import ModelGeometry

GB = 1e9
MTP_FLOOR_PER_LAYER = 76   # SlotStream's activation floor after head + context charges


@dataclass
class MemoryPlan:
    process_target_gb: float = 4.75   # 8 GB Mac: macOS + apps keep ~3 GB; Metal's
                                      # recommended working set is ~5.3 GB on 8 GB
    dense_gb: float = 1.50            # embeddings, attention, DeltaNet, shared experts, lm_head (4-bit)
    mtp_dense_gb: float = 0.06        # MTP block's dense part (its experts stream like any layer)
    router_dtype_bytes: int = 2       # fp16 on 8 GB (SlotStream keeps fp32: +~100 MB)
    context_tokens: int = 4096
    gdn_state_gb: float = 0.08        # all DeltaNet ssm+conv state, one copy (estimate)
    kv_bytes_per_token: int = 24_576  # 12 attn layers x 2 kv heads x 256 x (k,v) x fp16
    mlx_cache_cap_gb: float = 0.50    # mx.metal.set_cache_limit; SlotStream caps 2 GB on big Macs
    prefill_chunk: int = 256
    activation_mb_per_chunk_token: float = 0.6   # measure: SlotStream's model needs 1.30
    mtp_depth: int = 3
    spec_slot_cap: int = 24
    g: ModelGeometry = field(default_factory=ModelGeometry)

    def fixed_gb(self) -> Dict[str, float]:
        g = self.g
        routers = g.total_moe_layers * g.n_experts * g.hidden * self.router_dtype_bytes / GB
        recorded = self.gdn_state_gb * (self.mtp_depth + 1) if self.mtp_depth else 0.0
        return {
            "dense weights": self.dense_gb,
            "MTP dense": self.mtp_dense_gb if self.mtp_depth else 0.0,
            "routers (forecast)": routers,
            "DeltaNet state": self.gdn_state_gb,
            "recorded states (verify)": recorded,
            "KV cache": self.context_tokens * self.kv_bytes_per_token / GB,
            "MLX allocator cache cap": self.mlx_cache_cap_gb,
            "prefill activations": self.prefill_chunk * self.activation_mb_per_chunk_token / 1e3,
            "sweep staging (2x32)": 2 * 32 * g.slot_nbytes / GB,
        }

    @property
    def pool_gb(self) -> float:
        return self.process_target_gb - sum(self.fixed_gb().values())

    @property
    def slots(self) -> int:
        return int(self.pool_gb * GB // self.g.slot_nbytes)

    @property
    def experts_per_layer(self) -> float:
        return self.slots / self.g.total_moe_layers

    def validate(self) -> None:
        g = self.g
        if self.slots < 3 * g.top_k * 2:
            raise MemoryError(f"plan leaves {self.slots} slots: below a safe floor; refusing rather than swapping")

    def report(self) -> str:
        lines = [f"process target          {self.process_target_gb:5.2f} GB"]
        for k, v in self.fixed_gb().items():
            lines.append(f"  - {k:<22}{v:5.2f} GB")
        lines.append(f"expert pool             {self.pool_gb:5.2f} GB = {self.slots} slots "
                     f"= {self.experts_per_layer:.0f}/layer of {self.g.n_experts}")
        lines.append(f"MTP allowed (>= {MTP_FLOOR_PER_LAYER}/layer)? {mtp_allowed(self.experts_per_layer)}")
        return "\n".join(lines)


def mtp_allowed(experts_per_layer: float) -> bool:
    return experts_per_layer >= MTP_FLOOR_PER_LAYER


def decode_ceiling(hit_rate: float, disk_gbps: float, compute_ms: float,
                   g: ModelGeometry = ModelGeometry(), overlap: float = 0.0) -> float:
    """tok/s if each token reads its misses at `disk_gbps` effective and
    `overlap` of the IO hides behind compute. Uses main-model layers only."""
    uses = g.n_layers * g.top_k
    io_ms = uses * (1 - hit_rate) * g.slot_nbytes / (disk_gbps * GB) * 1e3
    step = compute_ms + io_ms * (1 - overlap) + max(0.0, io_ms * overlap - compute_ms)
    return 1e3 / step


def required_hit_rate(target_tok_s: float, disk_gbps: float, compute_ms: float,
                      g: ModelGeometry = ModelGeometry()) -> float:
    """Hit rate needed with zero overlap. > 1 means impossible."""
    budget_ms = 1e3 / target_tok_s - compute_ms
    if budget_ms <= 0:
        return float("inf")
    max_bytes = budget_ms / 1e3 * disk_gbps * GB
    miss = max_bytes / (g.n_layers * g.top_k * g.slot_nbytes)
    return 1 - miss


def main() -> None:
    plan = MemoryPlan()
    print("=== 8 GB memory plan (estimates; measure each line) ===")
    print(plan.report())
    plan_off = MemoryPlan(mtp_depth=0)
    print(f"\nwith MTP off: pool {plan_off.pool_gb:.2f} GB = {plan_off.experts_per_layer:.0f}/layer")

    g = plan.g
    print(f"\n=== bytes ===\nexpert record {g.slot_nbytes / 2**20:.2f} MiB; "
          f"{g.n_layers * g.top_k} expert uses per token; "
          f"all-miss token = {g.n_layers * g.top_k * g.slot_nbytes / 2**20:.0f} MiB")

    print("\n=== decode ceiling, tok/s (compute 40 ms/token assumed, no overlap) ===")
    disks = [1.5, 3.0, 6.0]
    print("hit rate   " + "".join(f"{d:>9.1f} GB/s" for d in disks))
    for h in [0.3, 0.45, 0.6, 0.75, 0.9]:
        print(f"  {h:4.2f}    " + "".join(f"{decode_ceiling(h, d, 40):>14.1f}" for d in disks))

    print("\n=== hit rate needed for 15 tok/s ===")
    for c in [0, 25, 40]:
        row = []
        for d in disks:
            r = required_hit_rate(15, d, c)
            row.append("impossible" if r == float("inf") else f"{r:.2f}")
        print(f"  compute {c:>2} ms: " + "  ".join(f"{d} GB/s -> {x}" for d, x in zip(disks, row)))


if __name__ == "__main__":
    main()
