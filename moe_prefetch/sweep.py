# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Scan-resistant prefill sweep. Mechanism adapted from SlotStream (MIT):
MoELayer.sweep / SlotPool.gatherResident / readStaged / admit, and the
"Prefill" notes in its CLAUDE.md.

A prompt pass of `min_tokens` or more never goes through the pool's demand
path. Default 16 here (SlotStream: 256). This only affects prompt passes,
never decode (T=1) or MTP verify passes. Below the threshold a prompt pass
pins and admits EVERY expert it routes: a 25-token chat prompt touches
~200 of 512 experts per layer, ~9,500 records, enough to churn a
1,600-slot pool several times and start decode on a cache shaped by the
prompt. Above it, the pool keeps its contents and only gains each layer's
fair share of the prompt's most-used experts. Note MLX's grouped
gather_qmm kernel needs >= 4 rows per expert to pay off, which a 16-token
pass rarely reaches, so short sweeps save cache churn, not compute. Per MoE layer:

  1. Route every row; sort the layer's routed experts by id and cut them
     into staging groups of 32.
  2. For each group, resident experts are COPIED OUT of the pool (reference
     bits untouched: a prompt says nothing about decode locality) and the
     rest are read from disk as contiguous runs, one vectored pread per run,
     at QD12 (throughput-bound).
  3. The group runs as one grouped GEMM over rows sorted by expert, so each
     expert's weights are read once per tile of tokens, not once per token.
     (MLX: gather_qmm with sorted_indices=True.)
  4. Reads of group g+1 overlap compute of group g: two staging buffers.

The sweep never writes the pool, so a long prompt cannot evict what decode
was using. Only the FINAL prompt pass admits, and only each layer's fair
share (slots / layers) of its most frequently routed experts.

Group composition depends on routing alone, never on residency, so the
arithmetic is identical on a cold or warm pool (SlotStream's sweep-check
invariant). Staging is bounded to two groups: 2 x 32 x 1.69 MiB ~= 108 MiB
for Qwen3-Next 4-bit, which matters at 8 GB.

What it cannot overlap: the router, attention and layer tail run while no
read is outstanding, because layer L+1's experts are unknown until its
router runs. SlotStream measured a background cross-layer read-ahead for
prefill slower in every paired run and removed it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .io_lanes import ExpertReader
from .pool import ByteSlabMemory, SlotPool, copy_record

GROUP = 32


@dataclass
class SweepStats:
    passes: int = 0
    layers: int = 0
    resident_copies: int = 0
    disk_records: int = 0
    admitted: int = 0
    read_wait_s: float = 0.0


class _Staging:
    """One staging group. `memory` has the same shape as pool memory, so the
    backend computes on it exactly as on the pool (rows = staging index)."""

    def __init__(self, memory):
        self.memory = memory
        self.index: Dict[int, int] = {}
        self.futures: List = []

    def views(self, e: int):
        return self.memory.views(self.index[e])


class PrefillSweep:
    def __init__(self, pool: SlotPool, reader: ExpertReader, min_tokens: int = 16,
                 group: int = GROUP, make_memory: Optional[Callable[[int], object]] = None):
        self.pool, self.reader = pool, reader
        self.min_tokens, self.group = min_tokens, group
        g = pool.g
        make_memory = make_memory or (lambda n: ByteSlabMemory(n, g.expert_nbytes, g.slot_nbytes))
        self._staging = [_Staging(make_memory(group)) for _ in range(2)]
        self.stats = SweepStats()

    def fair_share(self) -> int:
        return self.pool.slots // self.pool.g.total_moe_layers

    def _load(self, st: _Staging, layer: int, experts: List[int]) -> None:
        """Model thread: copy residents now, start disk runs for the rest."""
        st.index = {e: i for i, e in enumerate(experts)}      # sorted id order
        missing = []
        for e in experts:
            s = self.pool.lookup((layer, e))
            if s is not None:
                copy_record(st.views(e), self.pool.views(s))   # no ref-bit change
                self.stats.resident_copies += 1
            else:
                missing.append(e)
        self.stats.disk_records += len(missing)
        st.futures = self.reader.submit_runs(layer, missing, st.views)

    def run_layer(self, layer: int, x, ids: np.ndarray, w, moe: Callable, admit: bool):
        """moe(layer, x, ids, w, memory, rows) -> contribution of `rows`' experts only."""
        experts, counts = np.unique(ids, return_counts=True)
        experts = [int(e) for e in experts]
        groups = [experts[i:i + self.group] for i in range(0, len(experts), self.group)]
        admit_set = set()
        if admit:
            order = np.argsort(-counts, kind="stable")
            admit_set = {experts[i] for i in order[: self.fair_share()]}

        out = None
        self._load(self._staging[0], layer, groups[0])
        for gi, grp in enumerate(groups):
            st = self._staging[gi % 2]
            if gi + 1 < len(groups):                              # read ahead one group
                self._load(self._staging[(gi + 1) % 2], layer, groups[gi + 1])
            t0 = time.perf_counter()
            for f in st.futures:
                f.result()
            self.stats.read_wait_s += time.perf_counter() - t0
            part = moe(layer, x, ids, w, st.memory, dict(st.index))
            out = part if out is None else out + part
            if admit_set:
                members = [e for e in grp if e in admit_set]
                self.stats.admitted += self.pool.admit(
                    layer, members, len(members), lambda e, dst: copy_record(dst, st.views(e)))
        self.stats.layers += 1
        return out

    def close(self):
        for st in self._staging:
            st.memory.close()
