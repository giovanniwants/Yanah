# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Expert IO: QD32 demand batches, QD12 sweep runs, lane-budgeted speculation.
Structure follows SlotStream (MIT): ExpertStore.poolQueueDepth / defaultQueueDepth
and IOLaneBudget + slot adoption in Sources/Slotstream/ExpertPrefetch.swift.

Why two queue depths (SlotStream's measurements, M5 Pro):
* Decode misses are LATENCY-bound: a layer has a handful of records, and
  at QD12 the device queue empties between waves. QD32 measured 7.14 tok/s
  vs 6.82 at QD12 on 30 experts/layer; QD64 added nothing.
* Prefill sweep reads are THROUGHPUT-bound long contiguous runs: QD12 read
  179 tok/s, QD32 162 (oversubscribed), QD4 lost a third.

Speculation shares one lane budget with demand. While a demand batch is
being read, speculative workers finish only the read already in flight and
then wait, so a forecast can never delay a read the model is blocked on. A
ticket that a router demands while still queued is *promoted*: it takes a
lane immediately, over the speculative budget.

No asyncio here, deliberately. The demand path is one blocking batch per
layer (submit all, wait all), and the previous scaffold measured roughly
0.1-0.2 ms of event-loop hop overhead per request in Python, at ~490 records
per cold token. Plain threads + a condition variable do the same scheduling
with one wakeup per batch.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .layout import ExpertKey, ExpertLayout
from .pool import SlotPool

# Demand lanes. Raised 32 -> 64 on request. Expect no change in decode: a
# layer's demand batch holds at most T x top_k records (10 for one token,
# 40 for a 4-token verify pass), and this reader issues ONE vectored pread
# per record, so at most 10-40 reads are ever outstanding. SlotStream's
# QD32 gain came from nine preads per record (~90 per layer); it measured
# QD64 flat (7.16 vs 7.14 tok/s).
DEMAND_QD = 64
SWEEP_QD = 12


# ----------------------------------------------------------------- device
class SimulatedSSD:
    """Test-only device model: per-read latency pipelines across the queue,
    bytes share one bandwidth ceiling. Without the shared ceiling a QD32
    simulation would get 32x bandwidth for free."""

    def __init__(self, latency_s: float, gbps: float):
        self.latency, self.bw = latency_s, gbps * 1e9
        self._free_at = 0.0
        self._lock = threading.Lock()

    def charge(self, nbytes: int) -> None:
        now = time.perf_counter()
        with self._lock:
            self._free_at = max(self._free_at, now) + nbytes / self.bw
            done = max(now + self.latency, self._free_at)
        delay = done - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


@dataclass
class ReadStats:
    demand_batches: int = 0
    demand_records: int = 0
    sweep_runs: int = 0
    sweep_records: int = 0
    spec_records: int = 0
    bytes: int = 0
    demand_wait_s: float = 0.0
    sweep_wait_s: float = 0.0


class ExpertReader:
    def __init__(self, layout: ExpertLayout, demand_qd: int = DEMAND_QD,
                 sweep_qd: int = SWEEP_QD, device: Optional[SimulatedSSD] = None):
        self.layout = layout
        self.slot_bytes = layout.geometry.slot_nbytes
        self.device = device
        self._fds = [self._open(p) for p in layout.files]
        self._demand = cf.ThreadPoolExecutor(demand_qd, thread_name_prefix="demand-io")
        self._sweep = cf.ThreadPoolExecutor(sweep_qd, thread_name_prefix="sweep-io")
        self._pad = memoryview(bytearray(self.slot_bytes))  # write-only sink, contents ignored
        self.stats = ReadStats()

    @staticmethod
    def _open(path: str) -> int:
        fd = os.open(path, os.O_RDONLY)
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)  # no page-cache double copy
            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)  # random reads: no readahead
        return fd

    def _preadv(self, fd: int, bufs: List[memoryview], offset: int) -> int:
        total = sum(len(b) for b in bufs)
        done = 0
        while done < total:            # handle short reads across the vector
            skip, rest = done, []
            for b in bufs:
                if skip >= len(b):
                    skip -= len(b); continue
                rest.append(b[skip:]); skip = 0
            n = os.preadv(fd, rest, offset + done)
            if n <= 0:
                raise IOError(f"short read at {offset + done}")
            done += n
        if self.device:
            self.device.charge(total)
        self.stats.bytes += total
        return total

    def read_record(self, key: ExpertKey, dst: List[memoryview]) -> None:
        """One vectored pread: the record's pieces land in their own buffers."""
        loc = self.layout[key]
        if sum(len(d) for d in dst) != loc.nbytes:
            raise ValueError(f"{key}: buffers hold {sum(len(d) for d in dst)} B, record is {loc.nbytes} B")
        self._preadv(self._fds[loc.file_idx], list(dst), loc.offset)

    # ---- decode: one batch per layer, QD32, caller blocks ------------------
    def submit_batch(self, jobs: Sequence[Tuple[ExpertKey, List[memoryview]]]) -> List[cf.Future]:
        if jobs:
            self.stats.demand_batches += 1
            self.stats.demand_records += len(jobs)
        return [self._demand.submit(self.read_record, k, v) for k, v in jobs]

    def wait(self, futs: Sequence[cf.Future]) -> None:
        t0 = time.perf_counter()
        for f in futs:
            f.result()
        self.stats.demand_wait_s += time.perf_counter() - t0

    def read_batch(self, jobs: Sequence[Tuple[ExpertKey, List[memoryview]]]) -> None:
        self.wait(self.submit_batch(jobs))

    # ---- prefill sweep: contiguous runs, QD12 ------------------------------
    def runs(self, layer: int, experts: Sequence[int]) -> List[List[int]]:
        """Split sorted expert ids into runs contiguous on disk."""
        out: List[List[int]] = []
        for e in experts:
            if out:
                prev = self.layout[(layer, out[-1][-1])]
                cur = self.layout[(layer, e)]
                if cur.file_idx == prev.file_idx and cur.offset == prev.offset + self.slot_bytes:
                    out[-1].append(e); continue
            out.append([e])
        return out

    def submit_runs(self, layer: int, experts: Sequence[int],
                    dst_for: Callable[[int], List[memoryview]]) -> List[cf.Future]:
        """One vectored pread per contiguous run. Between records the file's
        alignment padding is read into a scratch sink (iovecs are free)."""
        futs = []
        for run in self.runs(layer, experts):
            first = self.layout[(layer, run[0])]
            bufs: List[memoryview] = []
            for i, e in enumerate(run):
                bufs += list(dst_for(e))
                pad = self.slot_bytes - self.layout[(layer, e)].nbytes
                if pad and i < len(run) - 1:
                    bufs.append(self._pad[:pad])
            futs.append(self._sweep.submit(self._preadv, self._fds[first.file_idx], bufs, first.offset))
            self.stats.sweep_runs += 1
            self.stats.sweep_records += len(run)
        return futs

    def close(self) -> None:
        self._demand.shutdown(wait=True)
        self._sweep.shutdown(wait=True)
        for fd in self._fds:
            os.close(fd)


# ------------------------------------------------------------ lane budget
class IOLaneBudget:
    """Direct port of SlotStream's IOLaneBudget."""

    def __init__(self, speculative_lanes: int):
        self.speculative_lanes = max(1, speculative_lanes)
        self._cv = threading.Condition()
        self._demand_active = 0
        self._spec_in_use = 0
        self.deferred = 0

    def begin_demand(self) -> None:
        with self._cv:
            self._demand_active += 1

    def end_demand(self) -> None:
        with self._cv:
            self._demand_active = max(0, self._demand_active - 1)
            self._cv.notify_all()

    def acquire_speculative(self, should_continue: Callable[[], bool],
                            priority: Callable[[], bool]) -> bool:
        with self._cv:
            waited = False
            while self._demand_active > 0 or self._spec_in_use >= self.speculative_lanes:
                if priority():
                    break
                if not should_continue():
                    return False
                waited = True
                self._cv.wait(0.005)
            if waited:
                self.deferred += 1
            self._spec_in_use += 1
            return True

    def release_speculative(self) -> None:
        with self._cv:
            self._spec_in_use = max(0, self._spec_in_use - 1)
            self._cv.notify_all()

    def wake(self) -> None:
        with self._cv:
            self._cv.notify_all()


# --------------------------------------------------- speculative tickets
class _Ticket:
    __slots__ = ("key", "slot", "target_layer", "generation", "state",
                 "cancelled", "promoted", "done", "lock")

    def __init__(self, key, slot, target_layer, generation):
        self.key, self.slot = key, slot
        self.target_layer, self.generation = target_layer, generation
        self.state = "queued"          # queued | reading | ready | failed | dropped
        self.cancelled = False
        self.promoted = False
        self.done = threading.Event()
        self.lock = threading.Lock()


@dataclass
class SpecStats:
    issued: int = 0
    below_threshold: int = 0
    adopted_ready: int = 0         # finished before demand: full win
    adopted_promoted: int = 0      # demand waited on an in-flight forecast
    expired: int = 0


class SpeculativeScheduler:
    """Router-forecast reads straight into reserved pool slots.

    Defaults are SlotStream's qualified decode bundle (0.2.19): forecast tap
    after the previous layer's attention, window 1 layer, top 24 per layer,
    probability threshold 0.062, issue cap 32 per target, slot adoption with
    a reservation cap. Lanes and slot cap are scaled down for 8 GB / base
    SSDs; SlotStream ships 16 lanes and a 64-slot cap on a 17 GB/s SSD."""

    def __init__(self, reader: ExpertReader, pool: SlotPool, budget: IOLaneBudget,
                 top_per_layer: int = 24, threshold: float = 0.062, issue_cap: int = 32):
        self.reader, self.pool, self.budget = reader, pool, budget
        self.top, self.threshold, self.issue_cap = top_per_layer, threshold, issue_cap
        self._workers = cf.ThreadPoolExecutor(budget.speculative_lanes + 2,
                                              thread_name_prefix="spec-io")
        self.tickets: Dict[ExpertKey, _Ticket] = {}
        self.stats = SpecStats()

    # model thread
    def issue(self, target_layer: int, candidates: Sequence[Tuple[int, float]],
              generation: int) -> int:
        ranked = sorted(candidates, key=lambda c: -c[1])[: self.top]
        n = 0
        for expert, p in ranked:
            if p < self.threshold:
                self.stats.below_threshold += 1
                continue
            if n >= self.issue_cap:
                break
            key = (target_layer, expert)
            if key in self.tickets or self.pool.is_resident(key):
                continue
            slot = self.pool.reserve_speculative(key)
            if slot is None:
                break                      # cap or headroom reached
            t = _Ticket(key, slot, target_layer, generation)
            self.tickets[key] = t
            self._workers.submit(self._work, t)
            n += 1
        self.stats.issued += n
        return n

    def _work(self, t: _Ticket) -> None:
        ok = self.budget.acquire_speculative(lambda: not t.cancelled, lambda: t.promoted)
        if not ok:
            self._finish(t, "dropped")
            return
        with t.lock:
            cancelled = t.cancelled
            if not cancelled:
                t.state = "reading"
        if cancelled:                      # _finish takes t.lock: call it outside
            self.budget.release_speculative()
            self._finish(t, "dropped")
            return
        try:
            self.reader.read_record(t.key, self.pool.views(t.slot))
            self.reader.stats.spec_records += 1
            state = "ready"
        except Exception:  # noqa: BLE001
            state = "failed"
        finally:
            self.budget.release_speculative()
        self._finish(t, state)

    def _finish(self, t: _Ticket, state: str) -> None:
        with t.lock:
            t.state = state
            give_back = t.cancelled or state != "ready"
        if give_back:
            self.pool.return_speculative(t.slot)
        t.done.set()

    def take(self, key: ExpertKey) -> Optional[int]:
        """Model thread, on a real routing decision. Returns a pinned slot
        holding the record, or None (caller does a demand read)."""
        t = self.tickets.pop(key, None)
        if t is None:
            return None
        if not t.done.is_set():
            t.promoted = True
            self.budget.wake()
            t.done.wait()
            promoted = True
        else:
            promoted = False
        if t.state != "ready" or t.cancelled:
            return None
        self.pool.adopt(t.slot, key)
        if promoted:
            self.stats.adopted_promoted += 1
        else:
            self.stats.adopted_ready += 1
        return t.slot

    def expire(self, generation: int, current_layer: int) -> None:
        """Drop forecasts whose target layer has passed. Their slots go back
        unused: a forecast that missed its layer is not cached."""
        stale = [k for k, t in self.tickets.items()
                 if t.generation < generation or t.target_layer < current_layer]
        for k in stale:
            t = self.tickets.pop(k)
            with t.lock:
                t.cancelled = True
                finished = t.state in ("ready", "failed", "dropped")
                ready = t.state == "ready"
            if finished and ready:
                self.pool.return_speculative(t.slot)
            self.stats.expired += 1
        if stale:
            self.budget.wake()

    def drain(self) -> None:
        pending = list(self.tickets.values())
        self.expire(1 << 62, 1 << 62)
        for t in pending:
            t.done.wait()
        self.pool.drain_returned()

    def close(self) -> None:
        self.drain()
        self._workers.shutdown(wait=True)
