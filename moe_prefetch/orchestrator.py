# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Forward-pass orchestrator. Owns the layer loop so it can route reads.

Two paths per MoE layer, chosen by pass size (as in SlotStream):

* POOL path (decode, MTP verify, short prompts):
      h_res, x, logits  = backend.pre_moe(l, h)
      ids, w, probs     = route_and_forecast(logits, h_res, [l+1 .. l+lookahead])
                          # ONE GPU sync: routing for l, forecast for the next layers
      submit demand(l)    # QD batch for l's misses starts reading
      issue forecasts     # queued while l's reads are in flight; the lane
                          # budget releases them the moment demand finishes
      finish demand(l)    # adopt ready forecasts, wait on the rest
      h = h_res + backend.moe(l, x, ids, w, slots); unpin

  With lookahead 2, layer l+2's experts get two layers of lead time. The
  forecast for l+2 uses l's post-attention state (SlotStream's stride-2
  tap). The state BEFORE l's attention is available earlier but forecasts
  worse at the same lead time (SlotStream: 0.62 vs 0.73 agreement), so
  extra lead comes from looking further ahead, not from an earlier tap.
  Forecasts read into the pool's speculative partition and join the cache
  only if the target layer's router really asks for them.

* SWEEP path (prompt passes >= sweep.min_tokens): see sweep.py.

Speculative verify passes run with recording on: the backend stores the
recurrent (Gated DeltaNet ssm + conv) state after every position, so a
rejection restores position n-1 and trims KV instead of re-running.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from .io_lanes import ExpertReader, IOLaneBudget, SpeculativeScheduler
from .layout import ExpertKey, ModelGeometry
from .pool import SlotPool
from .predictor import RouterForecast
from .sweep import PrefillSweep


class ModelBackend(Protocol):
    """What the MLX model exposes. Dense weights (embeddings, attention,
    DeltaNet, norms, routers, shared expert, lm_head) stay resident."""
    geometry: ModelGeometry

    def embed(self, tokens: np.ndarray) -> np.ndarray: ...
    def pre_moe(self, layer: int, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def moe(self, layer: int, x, ids: np.ndarray, w, memory, rows: Dict[int, int]):
        """Routed-expert contribution of the experts in `rows` only
        (expert id -> row of `memory`), so the sweep can call it once per
        staging group. `memory` is pool or staging memory. Anything the
        layer needs exactly once (shared expert) belongs in pre_moe."""

    # Optional hooks (used when present):
    #   route(logits) -> (ids np.ndarray (T,k), weights)   exact model routing
    #   forecast_input(h_res) -> np.ndarray (T,H) float32   for RouterForecast
    #   make_expert_memory(n) -> memory with views(i)/close()
    def lm_logits(self, h: np.ndarray) -> np.ndarray: ...
    def mtp_pre_moe(self, h_prev: np.ndarray, next_token: int): ...
    def mtp_logits(self, h: np.ndarray) -> np.ndarray: ...
    # State recording (SlotStream State.setRecording / rollback):
    def set_recording(self, on: bool) -> None: ...
    def rollback(self, keep: int) -> None:
        """After a recorded pass of T tokens keep the first `keep`: recurrent
        ssm AND conv states <- recorded[keep-1], KV trimmed to start+keep,
        recording cleared. keep == T just clears recording. No compute."""
    def mtp_reset(self) -> None: ...


@dataclass
class OrchStats:
    forwards: int = 0
    routed: int = 0
    resident_hits: int = 0
    adopted_ready: int = 0
    adopted_waited: int = 0
    demand_misses: int = 0
    seconds_by_T: Dict[int, List[float]] = field(default_factory=dict)
    # time breakdown, seconds, pool-path layers (see `breakdown`)
    t_route: float = 0.0       # routing sync: attention/GDN + router + forecast routers
                               # (+ the previous layer's experts when lazy)
    t_issue: float = 0.0       # CPU: forecast selection + issuing reads
    t_io: float = 0.0          # waiting for demand reads / promoted forecasts
    t_moe: float = 0.0         # experts: build + submit (+ GPU wait only when not lazy)
    t_forward: float = 0.0     # whole forward passes
    pool_layers: int = 0

    def summary(self) -> str:
        r = max(1, self.routed)
        return (f"routed={self.routed} resident={self.resident_hits / r:.1%} "
                f"forecast-ready={self.adopted_ready / r:.1%} forecast-waited={self.adopted_waited / r:.1%} "
                f"demand-miss={self.demand_misses / r:.1%}")

    def snapshot(self) -> "OrchStats":
        import copy
        return copy.deepcopy(self)

    def breakdown(self, since: Optional["OrchStats"] = None, per: int = 1) -> str:
        """ms per `per` (e.g. per decoded token) since a snapshot."""
        b = since or OrchStats()
        f = lambda a, c: (a - c) * 1e3 / max(1, per)
        parts = {"routing sync": f(self.t_route, b.t_route), "forecast": f(self.t_issue, b.t_issue),
                 "io wait": f(self.t_io, b.t_io), "experts": f(self.t_moe, b.t_moe)}
        total = f(self.t_forward, b.t_forward)
        parts["other (python, embed, lm_head)"] = total - sum(parts.values())
        return f"{total:.1f} ms = " + " + ".join(f"{k} {v:.1f}" for k, v in parts.items())


class MoEOrchestrator:
    def __init__(self, backend: ModelBackend, pool: SlotPool, reader: ExpertReader,
                 forecast: Optional[RouterForecast] = None,
                 scheduler: Optional[SpeculativeScheduler] = None,
                 sweep: Optional[PrefillSweep] = None, budget: Optional[IOLaneBudget] = None,
                 lookahead: int = 2, forecast_mtp: bool = False,
                 lazy_moe: bool = True, overlap_resident: bool = True):
        """lookahead: forecast layers l+1 .. l+lookahead at layer l.
        forecast_mtp: also forecast the MTP layer's experts (only useful
        while speculation is on; MTPSpeculator sets it).
        lazy_moe: don't block on each layer's experts; they finish inside the
          next layer's routing sync (needs backend.supports_lazy_moe).
        overlap_resident: in decode, start the experts already in memory (and
          the shared expert) on the GPU before waiting for the misses."""
        self.trace: Optional[list] = None   # set to [] to record decode routing (see cache_sim.py)
        self.lazy = lazy_moe and getattr(backend, "supports_lazy_moe", False)
        self.overlap = overlap_resident and self.lazy and hasattr(backend, "moe_ready")
        self.b, self.pool, self.reader = backend, pool, reader
        self.lookahead = max(0, lookahead)
        self.forecast_mtp = forecast_mtp
        self.fc, self.sched, self.sweep = forecast, scheduler, sweep
        self.budget = budget or (scheduler.budget if scheduler else IOLaneBudget(1))
        self.g = backend.geometry
        self.generation = 0
        self.stats = OrchStats()

    # ----------------------------------------------------------- routing
    def route(self, logits) -> Tuple[np.ndarray, np.ndarray]:
        """Backend routing if provided, else softmax / top-k / renormalise."""
        if hasattr(self.b, "route"):
            return self.b.route(logits)
        k = self.g.top_k
        z = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)
        ids = np.argpartition(-p, kth=k - 1, axis=-1)[:, :k]
        w = np.take_along_axis(p, ids, axis=-1)
        return ids, w / w.sum(axis=-1, keepdims=True)

    # ----------------------------------------------------- demand path
    def _fence(self) -> None:
        """No pool write may race a submitted graph that still reads slots."""
        if self.lazy:
            self.b.fence()

    def _submit_demand(self, layer: int, experts: List[int]):
        """Classify, claim CLOCK victims for misses and start their reads.
        Forecasts that already finished are adopted here, so they count as
        ready for the resident-first overlap; in-flight ones wait in finish."""
        self._fence()
        self.pool.drain_returned()
        out: Dict[int, int] = {}
        pending: List[int] = []
        misses: List[int] = []
        for e in experts:
            key = (layer, e)
            s = self.pool.hit(key)
            if s is not None:
                out[e] = s; self.stats.resident_hits += 1
                continue
            t = self.sched.tickets.get(key) if self.sched is not None else None
            if t is None:
                misses.append(e)
            elif t.done.is_set():
                s = self.sched.take(key)           # ready: adopt now, no wait
                if s is None:
                    misses.append(e)
                else:
                    out[e] = s; self.stats.adopted_ready += 1
            else:
                pending.append(e)
        jobs = []
        for e in misses:
            s = self.pool.claim_for_demand((layer, e))
            jobs.append(((layer, e), self.pool.views(s)))
            out[e] = s
        active = bool(jobs or pending)
        if active:
            self.budget.begin_demand()
        return out, pending, misses, self.reader.submit_batch(jobs), active

    def _finish_demand(self, layer: int, out, pending, misses, futs, active) -> Dict[int, int]:
        try:
            late = []
            for e in pending:
                s = self.sched.take((layer, e))     # promotes + waits if in flight
                if s is None:                       # forecast dropped/failed
                    s = self.pool.claim_for_demand((layer, e))
                    late.append(((layer, e), self.pool.views(s)))
                    misses.append(e)
                else:
                    self.stats.adopted_waited += 1
                out[e] = s
            futs = futs + self.reader.submit_batch(late)
            self.reader.wait(futs)
        finally:
            if active:
                self.budget.end_demand()
        for e in misses:
            self.pool.publish(out[e], (layer, e))
        self.stats.demand_misses += len(misses)
        return out

    def ensure(self, layer: int, experts: List[int]) -> Dict[int, int]:
        """Every expert resident and pinned; returns expert -> slot."""
        return self._finish_demand(layer, *self._submit_demand(layer, experts))

    def _targets(self, layer: int) -> List[int]:
        if self.sched is None or self.fc is None:
            return []
        limit = self.g.total_moe_layers if self.forecast_mtp else self.g.n_layers
        return [t for t in range(layer + 1, layer + 1 + self.lookahead) if t < limit]

    def _pool_layer(self, layer: int, h, pre):
        h_res, x, logits = pre
        t0 = time.perf_counter()
        targets = self._targets(layer)
        probs = None
        if targets and self.fc.backend_mode and hasattr(self.b, "route_and_forecast"):
            ids, w, probs = self.b.route_and_forecast(logits, h_res, targets)
        else:
            ids, w = self.route(logits)
        t1 = time.perf_counter()
        experts = [int(e) for e in np.unique(ids)]
        self.stats.routed += len(experts)
        if self.fc is not None:
            self.fc.observe(layer, ids)
        demand = self._submit_demand(layer, experts)
        t2 = time.perf_counter()
        # Resident-first overlap (decode): experts already in memory and the
        # shared expert start on the GPU now, while the misses are read.
        handle = None
        if self.overlap and len(ids) == 1:
            # Ready = hits + adopted forecasts. NOT the slots just claimed for
            # misses: they are in the map already but their reads are in flight.
            missing = set(demand[2])
            ready = {e: s for e, s in demand[0].items() if e not in missing}
            handle = self.b.moe_ready(layer, x, ids, self.pool.memory, ready)
        t3 = time.perf_counter()
        # Issue l+1 .. l+lookahead while l's reads are in flight. The lane
        # budget keeps them from starting until this layer's demand is done.
        fc_rec: Dict[int, list] = {}
        if self.sched is not None and targets:
            tap = None
            for t in targets:
                if probs is not None:
                    if t not in probs:
                        continue
                    cands = self.fc.select(t, probs[t], self.sched.top)
                else:
                    if tap is None:
                        tap = self.b.forecast_input(h_res) if hasattr(self.b, "forecast_input") else h_res
                    cands = self.fc.forecast(t, tap, self.sched.top)
                self.sched.issue(t, cands, self.generation)
                fc_rec[t] = cands
        if self.trace is not None and len(ids) == 1:
            # (forward pass id, layer, routed experts, {target layer: [(expert, p)]})
            self.trace.append((self.generation, layer, [int(e) for e in ids[0]], fc_rec))
        t4 = time.perf_counter()
        slots = self._finish_demand(layer, *demand)
        if self.sched is not None:
            # Only now: forecasts for THIS layer were adopted above; whatever
            # is left for layers <= l was wrong and goes back.
            self.sched.expire(self.generation, layer + 1)
        t5 = time.perf_counter()
        try:
            if handle is not None:
                out = self.b.moe_rest(layer, x, ids, w, self.pool.memory, slots, handle)
            elif self.lazy:
                out = self.b.moe(layer, x, ids, w, self.pool.memory, slots, sync=False)
            else:
                # The backend must have FINISHED reading the slots when it
                # returns (MLX: mx.eval), because unpinned slots can be overwritten.
                out = self.b.moe(layer, x, ids, w, self.pool.memory, slots)
            res = h_res + self._finalize(layer, out)
            if self.lazy:
                self.b.launch(res)       # GPU starts now; next routing sync completes it
        finally:
            for s in slots.values():
                self.pool.unpin(s)
        t6 = time.perf_counter()
        st = self.stats
        st.t_route += t1 - t0
        st.t_issue += t4 - t3
        st.t_io += (t2 - t1) + (t5 - t4)       # classify/claim/submit + waiting
        st.t_moe += (t3 - t2) + (t6 - t5)
        st.pool_layers += 1
        return res

    def _finalize(self, layer: int, routed):
        """Backend hook: e.g. add the shared expert exactly once, in the
        model's own summation order."""
        return self.b.moe_finalize(layer, routed) if hasattr(self.b, "moe_finalize") else routed

    # ----------------------------------------------------------- forward
    def forward(self, tokens, record: bool = False, admit: bool = True, batch: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Main model over T tokens -> (logits (T,V), hidden (T,H)).
        record=True for speculative verify; caller must then backend.rollback()."""
        t0 = time.perf_counter()
        tokens = np.asarray(tokens)
        T = len(tokens)
        self.generation += 1
        if self.sched is not None:
            self.sched.expire(self.generation, 0)
        # batch=True: the rows are B sequences decoding one token each (the
        # backend is in batch mode). Decode always uses the pool: each layer
        # reads the union of the rows' experts once, shared by every row.
        if batch and record:
            raise ValueError("speculative verify is not supported for a batch")
        use_sweep = (self.sweep is not None and not record and not batch
                     and T >= self.sweep.min_tokens)
        if record:
            self.b.set_recording(True)
        h = self.b.embed(tokens)
        for layer in range(self.g.n_layers):
            pre = self.b.pre_moe(layer, h)
            if use_sweep:
                self._fence()                    # the sweep copies from and admits into the pool
                h_res, x, logits = pre
                ids, w = self.route(logits)
                h = h_res + self._finalize(layer, self.sweep.run_layer(layer, x, ids, w, self.b.moe, admit))
            else:
                h = self._pool_layer(layer, h, pre)
        if use_sweep:
            self.sweep.stats.passes += 1
        logits = self.b.lm_logits(h)
        dt = time.perf_counter() - t0
        self.stats.forwards += 1
        self.stats.t_forward += dt
        self.stats.seconds_by_T.setdefault(T, []).append(dt)
        return logits, h

    def prefill(self, prompt, chunk: int):
        """Chunked prompt read; only the last pass admits into the pool.
        Returns (last logits, hidden of every prompt position)."""
        prompt = list(prompt)
        starts = list(range(0, len(prompt), chunk))
        hs = []
        for i, s in enumerate(starts):
            logits, h = self.forward(prompt[s:s + chunk], admit=(i == len(starts) - 1))
            hs.append(h)
        if len(hs) == 1:
            return logits, hs[0]
        if isinstance(hs[0], np.ndarray):
            return logits, np.concatenate(hs)
        import mlx.core as mx
        return logits, mx.concatenate(hs)

    def generate_batch(self, prompts, max_new, eos=(), prefill_chunk: int = 256,
                       step_log: Optional[list] = None, on_decode_start=None) -> List[List[int]]:
        """Greedy decode of several prompts as ONE batch, the way mlx-lm's
        BatchGenerator does it: prefill each prompt into its own cache, merge
        the caches, then one pass over B rows per step; finished sequences
        leave the batch (cache filter). max_new: int or one per prompt; the
        first token comes from the prompt pass, as in MTPSpeculator.generate.
        step_log, if given, receives (rows, seconds) per decode step."""
        b = self.b
        n = len(prompts)
        limits = [max_new] * n if isinstance(max_new, int) else list(max_new)
        eos = set(eos)
        caches, outs = [], []
        for p in prompts:
            b.reset()
            if self.fc is not None:
                self.fc.reset()
            logits, _ = self.prefill(list(p), prefill_chunk)
            outs.append([int(np.argmax(logits[-1]))])
            caches.append(b.take_cache())
        live = [i for i in range(n) if outs[i][-1] not in eos and len(outs[i]) < limits[i]]
        if not live:
            return outs
        if on_decode_start is not None:
            on_decode_start()
        b.begin_batch([caches[i] for i in live])
        try:
            while live:
                t0 = time.perf_counter()
                logits, _ = self.forward([outs[i][-1] for i in live], batch=True)
                if step_log is not None:
                    step_log.append((len(live), time.perf_counter() - t0))
                keep = []
                for r, i in enumerate(live):
                    tok = int(np.argmax(logits[r]))
                    outs[i].append(tok)
                    if tok not in eos and len(outs[i]) < limits[i]:
                        keep.append(r)
                if len(keep) < len(live):
                    if keep:
                        b.filter_batch(keep)
                    live = [live[r] for r in keep]
        finally:
            b.end_batch()
        return outs

    def mtp_forward(self, h_prev: np.ndarray, next_token: int) -> Tuple[np.ndarray, np.ndarray]:
        layer = self.g.n_layers
        if h_prev.ndim == 1:
            h_prev = h_prev[None]
        pre = self.b.mtp_pre_moe(h_prev, next_token)
        h = self._pool_layer(layer, h_prev, pre)
        return self.b.mtp_logits(h)[0], h[0]
