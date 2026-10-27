# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Offline expert-cache simulator for routing traces recorded on the real model
(`run_mlx_test.py --trace FILE`). Answers two questions from REAL routing
instead of guesses:

1. What do N extra pool slots buy? (e.g. RAM freed by a 4-bit KV cache)
   -> hit rate, demand misses and MiB read per token vs pool size.
2. Does a co-activation "semantic lattice" beat the router forecast?
   Experts are clustered per layer by how often they fire for the same
   token (spectral clustering of the co-activation matrix; expert ids are
   per layer, so groups are too). When a token "enters" a group (>= m of
   its experts in it), the group that most often follows it at each of the
   next 3 layers is prefetched whole.

Clustering and group transitions are fitted on the first prompts and every
policy is scored on held-out prompts. The pool is SlotPool's CLOCK with
the same speculative partition (prefetches never evict residents; an
adopted prefetch costs one CLOCK eviction, like the miss it replaced).

Assumption shared by all policies: a prefetch arrives in time. That is
optimistic for every prefetching policy, more so for ones that read more.
So "SSD MiB per token" is the number that keeps them honest on a disk-bound
engine: a policy that removes few misses but reads many more bytes loses.

    python -m moe_prefetch.cache_sim TRACE.npz [--disk-gbps 5] [--groups 20]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ trace
@dataclass
class Trace:
    routed: np.ndarray                                  # (S, L, k) decode steps x layers x top-k
    prompt: np.ndarray                                  # (S,) prompt index of each step
    fc: List[List[Dict[int, List[Tuple[int, float]]]]]  # [S][L] {target layer: [(expert, p)]}
    meta: dict


def save_trace(path: str, entries: Sequence, prompt_of_generation: Dict[int, int], meta: dict) -> None:
    """entries: MoEOrchestrator.trace items (generation, layer, ids, {target: [(e, p)]})."""
    L = meta["n_layers"]
    gens: List[int] = []
    by_gen: Dict[int, Dict[int, tuple]] = {}
    for gen, layer, ids, fc in entries:
        if layer >= L:
            continue                                     # MTP layer passes are not decode steps
        if gen not in by_gen:
            by_gen[gen] = {}
            gens.append(gen)
        by_gen[gen][layer] = (ids, fc)
    gens = [g for g in gens if len(by_gen[g]) == L and g in prompt_of_generation]   # complete decode passes
    k = len(by_gen[gens[0]][0][0])
    routed = np.zeros((len(gens), L, k), np.int16)
    fs, fl, ft, fe, fp = [], [], [], [], []
    for s, g in enumerate(gens):
        for l in range(L):
            ids, fc = by_gen[g][l]
            routed[s, l] = ids
            for t, cands in fc.items():
                for e, p in cands:
                    fs.append(s); fl.append(l); ft.append(t); fe.append(e); fp.append(p)
    np.savez_compressed(path, routed=routed,
                        prompt=np.array([prompt_of_generation[g] for g in gens], np.int32),
                        fc_step=np.array(fs, np.int32), fc_layer=np.array(fl, np.int16),
                        fc_target=np.array(ft, np.int16), fc_expert=np.array(fe, np.int16),
                        fc_prob=np.array(fp, np.float32), meta=json.dumps(meta))


def load_trace(path: str) -> Trace:
    z = np.load(path, allow_pickle=False)
    routed = z["routed"].astype(np.int64)
    S, L, _ = routed.shape
    fc: List[List[Dict[int, list]]] = [[{} for _ in range(L)] for _ in range(S)]
    for s, l, t, e, p in zip(z["fc_step"], z["fc_layer"], z["fc_target"], z["fc_expert"], z["fc_prob"]):
        fc[int(s)][int(l)].setdefault(int(t), []).append((int(e), float(p)))
    return Trace(routed, z["prompt"].astype(np.int64), fc, json.loads(str(z["meta"])))


# -------------------------------------------------------------- simulator
class PoolSim:
    """SlotPool semantics: CLOCK over demand slots + a separate partition
    of `spec_slots` for prefetches that are not yet used."""

    def __init__(self, slots: int, spec_slots: int):
        self.S = slots - spec_slots
        self.P = spec_slots
        self.key_of: List[Optional[tuple]] = [None] * self.S
        self.ref = [False] * self.S
        self.map: Dict[tuple, int] = {}
        self.free = list(range(self.S - 1, -1, -1))
        self.hand = 0
        self.staged: Dict[tuple, int] = {}               # prefetched, not yet used -> target layer
        self.measure = False
        self.c = dict(uses=0, hits=0, adopted=0, demand=0, prefetch=0, wasted=0, refused=0, tokens=0)
        self.miss_log: Optional[list] = None              # [(layer, [experts])] demand misses, if set

    def _count(self, name: str, n: int = 1) -> None:
        if self.measure:
            self.c[name] += n

    def _victim(self, pinned) -> int:
        if self.free:
            return self.free.pop()
        while True:
            s = self.hand
            self.hand = (s + 1) % self.S
            k = self.key_of[s]
            if k in pinned:
                continue
            if self.ref[s]:
                self.ref[s] = False
                continue
            del self.map[k]
            self.key_of[s] = None
            return s

    def _insert(self, key, pinned) -> None:
        s = self._victim(pinned)
        self.key_of[s] = key
        self.map[key] = s
        self.ref[s] = True

    def layer(self, l: int, experts: Sequence[int]) -> None:
        keys = [(l, int(e)) for e in set(experts)]
        pinned = set(keys)
        missed = []
        for key in keys:
            self._count("uses")
            s = self.map.get(key)
            if s is not None:
                self.ref[s] = True
                self._count("hits")
            elif key in self.staged:
                del self.staged[key]
                self._insert(key, pinned)
                self._count("adopted")
            else:
                self._insert(key, pinned)
                self._count("demand")
                missed.append(key[1])
        if missed and self.measure and self.miss_log is not None:
            self.miss_log.append((l, missed))

    def stage(self, key) -> Optional[bool]:
        """True staged, False skipped (already there), None partition full."""
        if key in self.map or key in self.staged:
            return False
        if len(self.staged) >= self.P:
            self._count("refused")
            return None
        self.staged[key] = key[0]
        self._count("prefetch")
        return True

    def expire(self, upto_layer: int) -> None:
        dead = [k for k, t in self.staged.items() if t <= upto_layer]
        for k in dead:
            del self.staged[k]
        self._count("wasted", len(dead))

    def end_token(self) -> None:
        self.expire(1 << 30)
        self._count("tokens")


# ---------------------------------------------------------------- policies
class ForecastPolicy:
    """Replays the engine's recorded router forecasts with its issue rules
    (probability >= threshold, at most issue_cap per target, stop when the
    partition is full)."""
    name = "router forecast"

    def __init__(self, trace: Trace, threshold: float = 0.062, issue_cap: int = 32):
        self.t, self.thr, self.cap = trace, threshold, issue_cap

    def issue(self, sim: PoolSim, s: int, l: int) -> None:
        for target, cands in self.t.fc[s][l].items():
            n = 0
            for e, p in sorted(cands, key=lambda c: -c[1]):
                if p < self.thr:
                    continue
                if n >= self.cap:
                    break
                r = sim.stage((target, e))
                if r is None:
                    return
                n += r


@dataclass
class Lattice:
    groups: List[np.ndarray]           # per layer: expert -> group (-1: never seen in training)
    members: List[List[np.ndarray]]    # per layer, per group: expert ids
    trans: Dict[int, List[np.ndarray]] # depth -> per layer: (G, G) counts group at l -> group at l+d
    enter_min: int
    depth: int


def _kmeans(X: np.ndarray, k: int, rng, iters: int = 60) -> np.ndarray:
    n = len(X)
    k = min(k, n)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):                                # k-means++
        d = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)])
    C = np.array(centers)
    lab = np.zeros(n, np.int64)
    for _ in range(iters):
        lab_new = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        if np.array_equal(lab_new, lab) and _ > 0:
            break
        lab = lab_new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def fit_lattice(trace: Trace, train: np.ndarray, n_experts: int, n_groups: int = 20,
                enter_min: int = 2, depth: int = 3, seed: int = 0) -> Lattice:
    rng = np.random.default_rng(seed)
    R = trace.routed[train]                              # (N, L, k)
    N, L, k = R.shape
    groups, members = [], []
    for l in range(L):
        X = np.zeros((N, n_experts), np.float32)
        np.put_along_axis(X, R[:, l, :], 1.0, axis=1)
        C = X.T @ X
        np.fill_diagonal(C, 0)
        deg = C.sum(1)
        act = np.where(deg > 0)[0]
        g = np.full(n_experts, -1, np.int64)
        if len(act) > n_groups:
            A = C[np.ix_(act, act)] / np.sqrt(np.outer(deg[act], deg[act]))
            w, V = np.linalg.eigh(A)
            emb = V[:, -n_groups:]
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            g[act] = _kmeans(emb, n_groups, rng)
        groups.append(g)
        members.append([np.where(g == j)[0] for j in range(n_groups)])
    trans = {}
    for d in range(1, depth + 1):
        per = []
        for l in range(L - d):
            gl, gt = groups[l][R[:, l, :]], groups[l + d][R[:, l + d, :]]
            cl = np.zeros((N, n_groups)); ct = np.zeros((N, n_groups))
            rows = np.repeat(np.arange(N), k)
            m = gl.ravel() >= 0
            np.add.at(cl, (rows[m], gl.ravel()[m]), 1)
            m = gt.ravel() >= 0
            np.add.at(ct, (rows[m], gt.ravel()[m]), 1)
            entered = (cl >= enter_min).astype(np.float64)
            has = ct.max(1) > 0
            dest = np.zeros((N, n_groups)); dest[np.arange(N)[has], ct[has].argmax(1)] = 1
            per.append(entered.T @ dest)
        trans[d] = per
    return Lattice(groups, members, trans, enter_min, depth)


class LatticePolicy:
    name = "semantic lattice"

    def __init__(self, lat: Lattice):
        self.lat = lat
        self.L = len(lat.groups)

    def issue(self, sim: PoolSim, s: int, l: int, experts: Sequence[int]) -> None:
        g = self.lat.groups[l][np.asarray(experts)]
        g = g[g >= 0]
        if not len(g):
            return
        cnt = np.bincount(g, minlength=len(self.lat.members[l]))
        entered = np.where(cnt >= self.lat.enter_min)[0]
        for d in range(1, self.lat.depth + 1):
            t = l + d
            if t >= self.L:
                break
            T = self.lat.trans[d][l]
            for g0 in entered:
                if T[g0].sum() <= 0:
                    continue
                for e in self.lat.members[t][int(T[g0].argmax())]:
                    if sim.stage((t, int(e))) is None:
                        return


# ------------------------------------------------------------------- runs
def simulate(trace: Trace, test: np.ndarray, slots: int, spec_slots: int,
             forecast: Optional[ForecastPolicy] = None,
             lattice: Optional[LatticePolicy] = None, miss_log: Optional[list] = None) -> dict:
    """Replay every step in order (earlier prompts warm the pool), count
    only on `test` steps. Engine order per layer: use l, issue, expire <= l."""
    sim = PoolSim(slots, spec_slots)
    sim.miss_log = miss_log
    S, L, _ = trace.routed.shape
    for s in range(S):
        sim.measure = bool(test[s])
        for l in range(L):
            ex = trace.routed[s, l]
            sim.layer(l, ex)
            if forecast is not None:
                forecast.issue(sim, s, l)
            if lattice is not None:
                lattice.issue(sim, s, l, ex)
            sim.expire(l)
        sim.end_token()
    c = dict(sim.c)
    n = max(1, c["tokens"])
    c["hit_rate"] = (c["hits"] + c["adopted"]) / max(1, c["uses"])
    c["demand_per_tok"] = c["demand"] / n
    c["prefetch_per_tok"] = c["prefetch"] / n
    c["useful_per_tok"] = c["adopted"] / n
    c["wasted_per_tok"] = c["wasted"] / n
    c["reads_per_tok"] = (c["demand"] + c["prefetch"]) / n
    return c


# ------------------------------------------------- co-activation repacking
def seriate(trace: Trace, train: np.ndarray, n_experts: int, n_groups: int = 20) -> list:
    """Per layer, a disk order that puts experts which fire together next to
    each other, fitted on training prompts: spectral clusters laid out
    contiguously, members ordered by the Fiedler vector of the normalized
    co-activation Laplacian. (A Fiedler-only 1-D order interleaves clusters;
    on planted groups this clustered order came within 1% of an oracle that
    knows the true groups.) Experts never seen in training go last."""
    lat = fit_lattice(trace, train, n_experts, n_groups, enter_min=2, depth=1)
    R = trace.routed[train]
    N, L, _ = R.shape
    orders = []
    for l in range(L):
        X = np.zeros((N, n_experts), np.float32)
        np.put_along_axis(X, R[:, l, :], 1.0, axis=1)
        C = X.T @ X
        np.fill_diagonal(C, 0)
        deg = C.sum(1)
        fied = np.zeros(n_experts)
        act = np.where(deg > 0)[0]
        if len(act) >= 3:
            d = 1 / np.sqrt(deg[act])
            w, V = np.linalg.eigh(np.eye(len(act)) - d[:, None] * C[np.ix_(act, act)] * d[None, :])
            fied[act] = V[:, 1]
        g = lat.groups[l].astype(np.float64)
        g[g < 0] = n_groups                              # unseen experts last
        orders.append(np.lexsort((np.arange(n_experts), fied, g)).astype(np.int64))
    return orders


def runs_per_layer(miss_log: list, pos: Optional[list]) -> float:
    """Reads needed if adjacent records merge into one read; pos[l][e] = disk position."""
    total = 0
    for l, ex in miss_log:
        p = np.sort(np.asarray(ex) if pos is None else pos[l][np.asarray(ex)])
        total += 1 + int((np.diff(p) != 1).sum())
    return total / max(1, len(miss_log))


def split(trace: Trace, train_frac: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    prompts = np.unique(trace.prompt)
    n_train = max(1, int(round(len(prompts) * train_frac)))
    if n_train >= len(prompts):
        raise SystemExit("need at least 2 prompts in the trace to hold some out")
    train = np.isin(trace.prompt, prompts[:n_train])
    return train, ~train


def report(trace: Trace, disk_gbps: float, n_groups: int, enter_min: int, depth: int,
           lattice_slots: Optional[int], extra_slots: Sequence[int]) -> dict:
    meta = trace.meta
    rec = meta["record_bytes"]
    slots, spec = meta["slots"], meta["spec_slots"]
    train, test = split(trace)
    S, L, k = trace.routed.shape
    print(f"trace: {S} decode steps ({test.sum()} held out) x {L} layers, top-{k}, "
          f"{len(np.unique(trace.prompt))} prompts | pool {slots} slots ({spec} speculative), "
          f"record {rec / 2**20:.2f} MiB, disk {disk_gbps:.2f} GB/s")
    mib = lambda r: r * rec / 2**20
    ssd_ms = lambda r: r * rec / (disk_gbps * 1e9) * 1e3

    def row(name, c):
        print(f"  {name:34s} hit {c['hit_rate']:6.1%} | demand misses {c['demand_per_tok']:6.1f}/tok "
              f"({ssd_ms(c['demand_per_tok']):5.1f} ms blocking) | prefetch {c['prefetch_per_tok']:6.1f}/tok "
              f"(useful {c['useful_per_tok']:5.1f}, wasted {c['wasted_per_tok']:5.1f}) | SSD "
              f"{mib(c['reads_per_tok']):6.0f} MiB/tok = {ssd_ms(c['reads_per_tok']):5.1f} ms busy")

    out = {}
    fpol = ForecastPolicy(trace)
    print("\n[1] pool size (current engine: CLOCK + router forecast)")
    base = None
    for d in [0] + list(extra_slots):
        c = simulate(trace, test, slots + d, spec, forecast=fpol)
        base = base or c
        row(f"{slots + d} slots ({'+' if d else ''}{d})" if d else f"{slots} slots (as recorded)", c)
        out[f"slots+{d}"] = c
    if "live_hit_rate" in meta:
        print(f"  calibration: the live run measured pool hit {meta['live_hit_rate']:.1%} "
              f"(all decode steps; the row above is held-out steps only)")

    print(f"\n[2] semantic lattice: {n_groups} groups per layer from {train.sum()} training steps, "
          f"enter >= {enter_min} experts, prefetch the likeliest group at the next {depth} layers")
    lat = fit_lattice(trace, train, meta["n_experts"], n_groups, enter_min, depth)
    sizes = [len(m) for m in lat.members[0] if len(m)]
    print(f"  layer-0 group sizes: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}")
    # How much of the next layers' routing do the predicted groups contain (held out)?
    lp = LatticePolicy(lat)
    cov, tot, size = 0, 0, 0
    for s in np.where(test)[0]:
        for l in range(L - 1):
            g = lat.groups[l][trace.routed[s, l]]
            g = g[g >= 0]
            if not len(g):
                continue
            entered = np.where(np.bincount(g, minlength=n_groups) >= enter_min)[0]
            pred = set()
            for g0 in entered:
                T = lat.trans[1][l]
                if T[g0].sum() > 0:
                    pred.update(int(e) for e in lat.members[l + 1][int(T[g0].argmax())])
            nxt = set(int(e) for e in trace.routed[s, l + 1])
            cov += len(nxt & pred); tot += len(nxt); size += len(pred)
    print(f"  next-layer coverage of predicted groups: {cov / max(1, tot):.1%} of routed experts, "
          f"from {size / max(1, test.sum() * (L - 1)):.1f} predicted experts per layer "
          f"(precision {cov / max(1, size):.1%})")
    lspec = lattice_slots or spec
    print(f"  (lattice runs with a {lspec}-slot speculative partition)")
    for name, kw, sp in [("cache only (no prefetch)", {}, spec),
                         ("router forecast (current)", dict(forecast=fpol), spec),
                         ("semantic lattice only", dict(lattice=lp), lspec),
                         ("router forecast + lattice", dict(forecast=fpol, lattice=lp), lspec)]:
        c = simulate(trace, test, slots, sp, **kw)
        row(name, c)
        out[name] = c
    print("\n[3] co-activation repacking: would a layer's demand misses sit next to each other?")
    log: list = []
    simulate(trace, test, slots, spec, forecast=fpol, miss_log=log)
    orders = seriate(trace, train, meta["n_experts"])
    pos = [np.argsort(o) for o in orders]               # expert -> disk position
    m = np.mean([len(ex) for _, ex in log]) if log else 0.0
    r_now, r_new = runs_per_layer(log, None), runs_per_layer(log, pos)
    print(f"  {len(log)} held-out layer steps with demand misses, {m:.2f} misses each on average")
    print(f"  reads per such layer: current order {r_now:.2f} | co-activation order {r_new:.2f} "
          f"({1 - r_new / max(1e-9, r_now):.0%} fewer reads; bytes unchanged)")
    print("  combine with disk_bench [2]: a read saved only helps if one adjacent read is faster "
          "than the same records read concurrently on this SSD")
    out["repack"] = dict(misses_per_layer=m, reads_now=r_now, reads_reordered=r_new)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--disk-gbps", type=float, default=None,
                    help="effective SSD bandwidth (default: the value measured in the traced run)")
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--enter-min", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lattice-slots", type=int, default=None,
                    help="speculative partition for lattice runs (default: same as recorded)")
    ap.add_argument("--extra-slots", default="20,41,82,327",
                    help="pool sizes to add (default: what 4-bit KV frees at 2K/4K/8K/32K context)")
    a = ap.parse_args()
    t = load_trace(a.trace)
    disk = a.disk_gbps or t.meta.get("disk_gbps") or 5.0
    report(t, disk, a.groups, a.enter_min, a.depth, a.lattice_slots,
           [int(x) for x in a.extra_slots.split(",") if x])


if __name__ == "__main__":
    main()
