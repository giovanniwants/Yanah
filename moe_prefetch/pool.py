# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Expert slot pool with CLOCK eviction. Mechanism adapted from SlotStream's
SlotPool (MIT, github.com/carloslfu/slotstream, Sources/Slotstream/ExpertStore.swift).

One fixed pool of uniform slots shared by every layer, carved out of a
single page-aligned anonymous mmap that is never resized.

* CLOCK: each slot has a reference bit. A hit sets it; the victim scan
  sweeps a hand around the pool, clears set bits, and evicts the first slot
  whose bit is clear and that is neither pinned nor reserved. SlotStream
  measured LRU-style and route-memo alternatives and kept CLOCK.
* Pins: the experts of the layer being computed are pinned until its
  compute has consumed them, so no victim scan can overwrite them.
* Speculative partition ("slot adoption"): `spec_slot_cap` slots belong to
  speculation and are invisible to every victim scan. A forecast reads
  straight into one of them and stays OUT of the lookup map. A forecast
  that is never demanded goes back to the partition: an unused forecast
  can never evict a resident expert. A forecast that IS demanded is adopted
  in place (bookkeeping, no copy) and joins the cache; the partition then
  takes one CLOCK victim to stay full. That is the eviction the demand
  miss would have caused anyway, so a correct forecast changes WHEN the
  eviction happens, not how many there are.
  (Why not "never evict, only recycle unused forecasts": after warm-up the
  pool has no free slots and every adoption keeps its slot, so the
  recyclable set shrinks by one per correct forecast and prefetching stops.)
* Scan resistance: the prefill sweep reads resident experts out of the pool
  without touching reference bits and never writes the pool; only the
  final prompt pass admits a per-layer fair share via `admit`.

Threading: only the model thread mutates pool state. Speculative workers
write slot bytes and hand slots back through a locked return queue that
the model thread drains (as SlotStream's drainReturnedSlots does).
"""
from __future__ import annotations

import mmap
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .layout import ExpertKey, ModelGeometry


@dataclass
class PoolStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    adopted: int = 0              # speculative records that a router demanded
    spec_reserved: int = 0
    spec_discarded: int = 0       # read but never demanded
    spec_refused: int = 0         # partition empty
    adopt_evictions: int = 0      # CLOCK victims taken to refill the partition after adoption
    admitted: int = 0             # prefill admissions

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.hits + self.misses)


class ByteSlabMemory:
    """Default slot memory: one page-aligned anonymous mmap, one buffer per
    record. The MLX backend supplies its own memory instead (one array per
    weight piece) so reads land directly in what gather_qmm indexes."""

    def __init__(self, n: int, record_bytes: int, slot_bytes: int):
        self.n, self.record_bytes, self.slot_bytes = n, record_bytes, slot_bytes
        self._buf = mmap.mmap(-1, max(1, n * slot_bytes), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self._view = memoryview(self._buf)

    def views(self, i: int) -> List[memoryview]:
        s = i * self.slot_bytes
        return [self._view[s:s + self.record_bytes]]

    def base_address(self) -> int:
        import ctypes
        return ctypes.addressof(ctypes.c_char.from_buffer(self._buf))

    def close(self) -> None:
        self._view.release()
        self._buf.close()


def copy_record(dst: List[memoryview], src: List[memoryview]) -> None:
    for d, s in zip(dst, src):
        d[:] = s


class SlotPool:
    def __init__(self, geometry: ModelGeometry, slots: int,
                 spec_slot_cap: int = 32, spec_headroom: Optional[int] = None,
                 memory=None):
        # spec_headroom is accepted for compatibility; the partition replaces it.
        self.g = geometry
        self.slot_bytes = geometry.slot_nbytes
        self.slots = slots
        if slots < 3 * geometry.top_k:
            raise ValueError(f"{slots} slots cannot hold a layer's routing plus margin")
        self.memory = memory if memory is not None else ByteSlabMemory(
            slots, geometry.expert_nbytes, self.slot_bytes)
        self.key_of: List[Optional[ExpertKey]] = [None] * slots
        self.ref_bit = [False] * slots
        self.pins = [0] * slots
        self.in_flight = [False] * slots       # reserved for a speculative read
        self.map: Dict[ExpertKey, int] = {}
        self.free: List[int] = list(range(slots - 1, -1, -1))
        self.hand = 0
        self.pinned_count = 0
        self.in_flight_count = 0
        self.spec_slot_cap = spec_slot_cap
        if slots - spec_slot_cap < 3 * geometry.top_k:
            raise ValueError(f"spec_slot_cap {spec_slot_cap} leaves too few demand slots in {slots}")
        self.spec_owned = [False] * slots       # partition membership (idle or in flight)
        self.spec_free: List[int] = []
        for _ in range(spec_slot_cap):
            s = self.free.pop()
            self.spec_owned[s] = True
            self.spec_free.append(s)
        self._spec_deficit = 0                  # adoptions that could not take a victim yet
        # SlotStream refuses reservations unless >256 slots stay scannable on
        # a pool of thousands. Scaled for small pools: two layers' routing.
        self.spec_headroom = spec_headroom if spec_headroom is not None else 2 * geometry.top_k + 8
        self._returned: List[int] = []
        self._returned_lock = threading.Lock()
        self.stats = PoolStats()

    # --------------------------------------------------------------- memory
    def views(self, slot: int) -> List[memoryview]:
        """Writable buffers for one record, in on-disk piece order."""
        return self.memory.views(slot)

    def lookup(self, key: ExpertKey) -> Optional[int]:
        return self.map.get(key)

    def is_resident(self, key: ExpertKey) -> bool:
        return key in self.map

    # ---------------------------------------------------------------- CLOCK
    def _victim(self) -> int:
        if self.free:
            return self.free.pop()
        scanned = 0
        while True:
            s = self.hand
            self.hand = (self.hand + 1) % self.slots
            scanned += 1
            if scanned > 3 * self.slots:
                raise MemoryError(f"slot pool exhausted: all {self.slots} slots pinned or reserved")
            if self.pins[s] or self.in_flight[s] or self.spec_owned[s]:
                continue
            if self.ref_bit[s]:
                self.ref_bit[s] = False
                continue
            old = self.key_of[s]
            if old is not None:
                del self.map[old]
                self.key_of[s] = None
                self.stats.evictions += 1
            return s

    # ------------------------------------------------------------ demand API
    def pin(self, slot: int) -> None:
        if self.pins[slot] == 0:
            self.pinned_count += 1
        self.pins[slot] += 1

    def unpin(self, slot: int) -> None:
        self.pins[slot] -= 1
        if self.pins[slot] == 0:
            self.pinned_count -= 1

    def hit(self, key: ExpertKey) -> Optional[int]:
        """Resident: set the reference bit, pin, return slot."""
        s = self.map.get(key)
        if s is None:
            return None
        self.ref_bit[s] = True
        self.pin(s)
        self.stats.hits += 1
        return s

    def claim_for_demand(self, key: ExpertKey) -> int:
        """Choose a CLOCK victim for a demand miss and pin it. The caller
        reads into it, then calls `publish`."""
        s = self._victim()
        self.pin(s)
        self.stats.misses += 1
        return s

    def publish(self, slot: int, key: ExpertKey) -> None:
        self.key_of[slot] = key
        self.map[key] = slot
        self.ref_bit[slot] = True

    def abandon(self, slot: int) -> None:
        """Demand read failed: the slot holds garbage, give it back."""
        self.unpin(slot)
        self.free.append(slot)

    # ------------------------------------------------------ speculative API
    def reserve_speculative(self, key: ExpertKey) -> Optional[int]:
        """Take an idle partition slot for a forecast read. Never evicts."""
        self.drain_returned()
        if key in self.map:
            return None
        if not self.spec_free:
            self._refill_partition()
        if not self.spec_free:
            self.stats.spec_refused += 1
            return None
        s = self.spec_free.pop()
        self.in_flight[s] = True
        self.in_flight_count += 1
        self.stats.spec_reserved += 1
        return s

    def _refill_partition(self) -> None:
        while self._spec_deficit:
            try:
                v = self._victim()
            except MemoryError:
                return                            # everything pinned right now; retry later
            self.spec_owned[v] = True
            self.ref_bit[v] = False
            self.spec_free.append(v)
            self._spec_deficit -= 1
            self.stats.adopt_evictions += 1

    def adopt(self, slot: int, key: ExpertKey) -> None:
        """A router demanded a completed forecast: publish + pin, no copy."""
        assert self.in_flight[slot] and self.key_of[slot] is None and self.spec_owned[slot]
        self.in_flight[slot] = False
        self.in_flight_count -= 1
        self.spec_owned[slot] = False             # leaves the partition, joins the cache
        self.publish(slot, key)
        self.pin(slot)
        self.stats.adopted += 1
        self.stats.hits += 1   # it was ready when demanded: counts as a hit
        self._spec_deficit += 1
        self._refill_partition()                  # one CLOCK victim, as the miss would have cost

    def return_speculative(self, slot: int) -> None:
        """Any thread: forecast cancelled or never demanded."""
        with self._returned_lock:
            self._returned.append(slot)

    def drain_returned(self) -> None:
        with self._returned_lock:
            drained, self._returned = self._returned, []
        for s in drained:
            if self.in_flight[s]:
                self.in_flight[s] = False
                self.in_flight_count -= 1
                self.stats.spec_discarded += 1
                self.spec_free.append(s)          # back to the partition; no resident touched

    # ------------------------------------------------- prefill (scan-safe)
    def admit(self, layer: int, experts_by_frequency: Sequence[int], share: int,
              copy_into: "callable") -> int:
        """Admit up to `share` of a layer's most-used prompt experts.
        `copy_into(expert, dst_view)` fills the slot from sweep staging.
        Resident ones are just marked referenced. Returns records written."""
        written = 0
        for e in list(experts_by_frequency)[:share]:
            key = (layer, e)
            s = self.map.get(key)
            if s is not None:
                self.ref_bit[s] = True
                continue
            s = self._victim()
            copy_into(e, self.views(s))
            self.publish(s, key)
            written += 1
        self.stats.admitted += written
        return written

    def resident_count(self) -> int:
        return len(self.map)

    def close(self) -> None:
        self.memory.close()
