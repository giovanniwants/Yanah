# SPDX-License-Identifier: FSL-1.1-ALv2
"""
End-to-end checks on a toy MoE written to disk, with a simulated SSD
(per-read latency pipelined across the queue, one shared bandwidth cap).

  1. Pool-path forward == reference forward reading experts directly.
  2. Prefill sweep: matches the pool path; bit-identical on a cold or warm
     pool; non-final passes leave pool contents AND CLOCK bits untouched;
     the final pass admits at most each layer's fair share.
  3. Decode: QD1 vs QD32 demand, forecasts off vs on, identical output.
  4. MTP: fixed depth 3 and adaptive policy produce exactly the greedy
     output; recorded-state rollback leaves the same state as plain decode.

Run:  python -m moe_prefetch.test_synthetic
"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from moe_prefetch import (ExpertLayout, ExpertReader, IOLaneBudget, ModelGeometry,
                          MoEOrchestrator, MTPSpeculator, PrefillSweep, RouterForecast,
                          SimulatedSSD, SlotPool, SpeculativeScheduler)
from moe_prefetch.predictor import rms_norm

G = ModelGeometry(n_layers=8, n_mtp_layers=1, n_experts=64, top_k=4,
                  hidden=64, moe_inter=32, bits=16, group_size=1, scale_bytes=0)
V = 97
CONV = 3   # short-conv window, like Gated DeltaNet's conv1d


def write_experts(path, rng):
    H, I = G.hidden, G.moe_inter
    with open(path, "wb") as f:
        for _ in range(G.total_moe_layers * G.n_experts):
            blob = np.concatenate([
                (rng.standard_normal((I, H)) / np.sqrt(H)).ravel(),
                (rng.standard_normal((I, H)) / np.sqrt(H)).ravel(),
                (rng.standard_normal((H, I)) / np.sqrt(I)).ravel()]).astype(np.float16).tobytes()
            f.write(blob + b"\0" * (G.slot_nbytes - len(blob)))


def expert_mlp(blob, x):
    H, I = G.hidden, G.moe_inter
    w = blob[:3 * H * I].astype(np.float32)
    gate, up, down = w[:I * H].reshape(I, H), w[I * H:2 * I * H].reshape(I, H), w[2 * I * H:].reshape(H, I)
    a = x @ gate.T
    return ((a / (1 + np.exp(-a))) * (x @ up.T)) @ down.T


class ToyBackend:
    """Recurrent mixer with an ssm state AND a conv window, both recorded
    per position during verify, like SlotStream's LinearCache."""

    def __init__(self, seed=1, compute_s=0.0):
        rng = np.random.default_rng(seed)
        self.geometry = G
        self.compute_s = compute_s
        H, E, L = G.hidden, G.n_experts, G.total_moe_layers
        self.emb = rng.standard_normal((V, H)).astype(np.float32)
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True) / 4
        succ = rng.permutation(V)
        self.w_out = np.zeros((V, H), np.float32); self.w_out[succ] = self.emb
        base = rng.standard_normal((E, H)).astype(np.float32)
        self.gates = [((base + 0.5 * rng.standard_normal((E, H))) * 2).astype(np.float32) for _ in range(L)]
        self.mix = [rng.standard_normal((H, H)).astype(np.float32) * 0.05 for _ in range(L)]
        self.mtp_proj = np.concatenate([np.eye(H), 0.1 * rng.standard_normal((H, H))]).astype(np.float32)
        self.ssm = [np.zeros(H, np.float32) for _ in range(G.n_layers)]
        self.conv = [np.zeros((CONV, H), np.float32) for _ in range(G.n_layers)]
        self.recording = False
        self.rec = None
        self.fed = 0          # positions consumed by the state

    def embed(self, toks):
        return self.emb[toks]

    def pre_moe(self, layer, h):
        out = np.empty_like(h)
        if layer == 0:
            self.fed += h.shape[0]
        s, c = self.ssm[layer], self.conv[layer]
        for t in range(h.shape[0]):
            c = np.vstack([c[1:], h[t:t + 1]])
            s = 0.8 * s + 0.2 * c.mean(0)
            if self.recording:
                self.rec[layer].append((s.copy(), c.copy()))
            out[t] = h[t] + self.mix[layer] @ s
        self.ssm[layer], self.conv[layer] = s, c
        x = rms_norm(out, None)
        return out, x, x @ self.gates[layer].T

    def moe(self, layer, x, ids, w, memory, rows):
        out = np.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                if e in rows:
                    blob = np.frombuffer(memory.views(rows[e])[0], dtype=np.float16)
                    out[t] += w[t, j] * expert_mlp(blob, x[t:t + 1])[0]
        if self.compute_s:
            time.sleep(self.compute_s)
        return 0.3 * out

    def lm_logits(self, h):
        return rms_norm(h, None) @ self.w_out.T * 4

    def mtp_pre_moe(self, h_prev, tok):
        z = np.concatenate([rms_norm(self.emb[[tok]], None), 0.2 * rms_norm(h_prev, None)], 1) @ self.mtp_proj
        x = rms_norm(z, None)
        return z, x, x @ self.gates[G.n_layers].T

    def mtp_logits(self, h):
        return self.lm_logits(h)

    def set_recording(self, on):
        self.recording = on
        self.rec = [[] for _ in range(G.n_layers)] if on else None

    def rollback(self, keep):
        self.fed -= len(self.rec[0]) - keep
        for l in range(G.n_layers):
            self.ssm[l], self.conv[l] = (a.copy() for a in self.rec[l][keep - 1])
        self.set_recording(False)

    def mtp_reset(self):
        pass

    def state(self):
        return np.concatenate([np.concatenate([s, c.ravel()]) for s, c in zip(self.ssm, self.conv)])


def reference_logits(b, path, tokens):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    o = MoEOrchestrator(b, None, None)
    h = b.embed(np.asarray(tokens))
    for l in range(G.n_layers):
        h_res, x, rl = b.pre_moe(l, h)
        ids, w = o.route(rl)
        class Direct:
            def views(self, e):
                off = (l * G.n_experts + int(e)) * G.slot_nbytes
                return [memoryview(np.ascontiguousarray(mm[off:off + G.expert_nbytes]))]
        rows = {int(e): int(e) for e in np.unique(ids)}
        h = h_res + b.moe(l, x, ids, w, Direct(), rows)
    return b.lm_logits(h)


def build(path, backend, slots=80, demand_qd=32, forecasts=True, sweep_min=32,
          device=None, lanes=4, spec_cap=12, lookahead=2):
    layout = ExpertLayout.contiguous(G, path)
    pool = SlotPool(G, slots, spec_slot_cap=spec_cap)
    reader = ExpertReader(layout, demand_qd=demand_qd, sweep_qd=12, device=device)
    budget = IOLaneBudget(lanes)
    fc = RouterForecast(G, backend.gates) if forecasts else None
    sched = SpeculativeScheduler(reader, pool, budget, top_per_layer=8, threshold=0.062) if forecasts else None
    sweep = PrefillSweep(pool, reader, min_tokens=sweep_min) if sweep_min else None
    orch = MoEOrchestrator(backend, pool, reader, fc, sched, sweep, budget, lookahead=lookahead)

    def close():
        if sched: sched.close()
        if sweep: sweep.close()
        reader.close(); pool.close()
    return orch, close


def main():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "experts.bin")
    write_experts(path, np.random.default_rng(0))
    rng = np.random.default_rng(7)
    long_prompt = [int(t) for t in rng.integers(0, V, 70)]
    prompt = [3, 14, 15, 92, 65]

    # 1 ---------------------------------------------------------------
    b1, b2 = ToyBackend(), ToyBackend()
    orch, close = build(path, b1, sweep_min=0)
    got, _ = orch.forward(prompt)
    ref = reference_logits(b2, path, prompt)
    assert np.allclose(got, ref, atol=1e-4)
    close()
    print("[1] pool-path forward matches reference")

    # 2 ---------------------------------------------------------------
    b_pool, b_sw = ToyBackend(), ToyBackend()
    o_pool, c1 = build(path, b_pool, sweep_min=0, slots=600, forecasts=False)
    o_sw, c2 = build(path, b_sw, sweep_min=32, slots=80, forecasts=False)
    lp, _ = o_pool.forward(long_prompt[:64])
    pool = o_sw.pool
    before = (dict(pool.map), list(pool.ref_bit))
    ls, _ = o_sw.forward(long_prompt[:64], admit=False)
    assert (dict(pool.map), list(pool.ref_bit)) == before, "non-final sweep touched the pool"
    # different summation order (per-group accumulation), as SlotStream notes
    assert np.allclose(lp, ls, rtol=1e-5, atol=1e-3)
    o_cold, c3 = build(path, ToyBackend(), sweep_min=32, slots=80, forecasts=False)
    cold, _ = o_cold.forward(long_prompt[:64], admit=False)
    o_warm, c4 = build(path, ToyBackend(), sweep_min=32, slots=80, forecasts=False)
    o_warm.forward(long_prompt[:64], admit=True)
    resident = o_warm.pool.resident_count()
    per_layer = {}
    for (l, e) in o_warm.pool.map:
        per_layer[l] = per_layer.get(l, 0) + 1
    o_warm.b.__init__()                       # fresh model state, warm pool
    warm, _ = o_warm.forward(long_prompt[:64], admit=False)
    assert np.array_equal(cold, warm), "sweep arithmetic depends on residency"
    share = o_warm.sweep.fair_share()
    assert max(per_layer.values()) <= share
    print(f"[2] sweep == pool path; bit-identical cold vs warm ({resident} resident); "
          f"non-final pass left pool and CLOCK bits untouched; final admit <= {share}/layer")
    print("    warm sweep:", o_warm.sweep.stats)
    for c in (c1, c2, c3, c4):
        c()

    # 3 ---------------------------------------------------------------
    def dev():  # scaled real ratios: record read ~0.5 ms, compute ~0.8 ms/layer
        return SimulatedSSD(latency_s=0.0001, gbps=G.slot_nbytes / 0.0005 / 1e9)
    results = {}
    for name, kw in [("QD1, no forecast", dict(demand_qd=1, forecasts=False)),
                     ("QD32, no forecast", dict(demand_qd=32, forecasts=False)),
                     ("QD32 + lookahead 1", dict(demand_qd=32, forecasts=True, lookahead=1)),
                     ("QD32 + lookahead 2", dict(demand_qd=32, forecasts=True, lookahead=2))]:
        b = ToyBackend(compute_s=0.0008)
        orch, close = build(path, b, device=dev(), **kw)
        spec = MTPSpeculator(orch, enabled=False, prefill_chunk=32)
        t0 = time.perf_counter()
        toks = spec.generate(long_prompt, 48)
        dt = time.perf_counter() - t0
        results[name] = toks
        print(f"[3] {name:18s} {len(toks) / dt:6.1f} tok/s | {orch.stats.summary()} "
              f"| pool hit {orch.pool.stats.hit_rate:.0%}")
        if orch.sched:
            s = orch.sched.stats
            print(f"    forecasts issued={s.issued} expired={s.expired}; recall={orch.fc.stats.recall:.0%}; "
                  f"spec reads deferred behind demand={orch.budget.deferred}; "
                  f"slots discarded unused={orch.pool.stats.spec_discarded}")
        close()
    assert len({tuple(v) for v in results.values()}) == 1, "prefetch changed output"

    # 4 ---------------------------------------------------------------
    def plain_run(n):
        b = ToyBackend(compute_s=0.0008)
        orch, close = build(path, b, device=dev())
        toks = MTPSpeculator(orch, enabled=False, prefill_chunk=32).generate(long_prompt, n)
        close()
        return toks, b
    plain, _ = plain_run(40)
    for name, kw in [("MTP fixed depth 3", dict(max_depth=3, adaptive=False)),
                     ("MTP adaptive (max 3)", dict(max_depth=3, adaptive=True))]:
        b = ToyBackend(compute_s=0.0008)
        orch, close = build(path, b, device=dev())
        sp = MTPSpeculator(orch, prefill_chunk=32, **kw)
        t0 = time.perf_counter()
        toks = sp.generate(long_prompt, 40)
        dt = time.perf_counter() - t0
        assert toks == plain, (name, toks, plain)
        print(f"[4] {name:21s} {len(toks) / dt:6.1f} tok/s, identical to greedy | {sp.stats.summary()}")
        close()
        # Same number of consumed positions via plain decode -> same state?
        _, pb = plain_run(b.fed - len(long_prompt) + 1)
        assert pb.fed == b.fed
        assert np.allclose(b.state(), pb.state(), atol=1e-5), "rollback left a different state"
        print(f"    recorded-state rollback: recurrent state after {b.fed} positions matches plain decode")
    # 5 ---------------------------------------------------------------
    # Pool invariant: an unused forecast never evicts a resident expert, and
    # the partition keeps prefetching alive after warm-up.
    pool = SlotPool(G, 40, spec_slot_cap=8)
    for e in range(32):                                   # fill every demand slot
        s = pool.claim_for_demand((0, e)); pool.publish(s, (0, e)); pool.unpin(s)
    before = dict(pool.map)
    for rnd in range(5):                                  # 5 rounds of wrong forecasts
        got = [pool.reserve_speculative((1, e)) for e in range(rnd * 8, rnd * 8 + 8)]
        assert all(g is not None for g in got), "partition starved by unused forecasts"
        for g_ in got:
            pool.return_speculative(g_)
        pool.drain_returned()
    assert pool.map == before and pool.stats.evictions == 0, "an unused forecast evicted a resident"
    s = pool.reserve_speculative((2, 0)); pool.adopt(s, (2, 0)); pool.unpin(s)
    assert pool.stats.adopt_evictions == 1 and len(pool.spec_free) == 8
    print("[5] 40 unused forecasts evicted 0 residents; an adopted forecast cost exactly 1 CLOCK "
          "eviction and the partition stayed full")

    # Ablation: the literal rule (recycle only unused forecasts, never refill)
    def run_issued(literal):
        b = ToyBackend(compute_s=0.0)
        orch, close = build(path, b, device=dev())
        if literal:
            orch.pool._refill_partition = lambda: None   # adopted slots are never replaced
        spec = MTPSpeculator(orch, enabled=False, prefill_chunk=32)
        counts = []
        orig = orch.sched.issue
        def counting(*a, **k):
            n = orig(*a, **k); counts.append(n); return n
        orch.sched.issue = counting
        spec.generate(long_prompt, 48)
        close()
        q = len(counts) // 4
        return sum(counts[:q]), sum(counts[-q:])
    lit, part = run_issued(True), run_issued(False)
    print(f"    forecasts issued, first vs last quarter of decode: literal rule {lit[0]} -> {lit[1]}, "
          f"partition {part[0]} -> {part[1]}")
    assert lit[1] < part[1]

    print("\nall checks passed")


if __name__ == "__main__":
    main()
