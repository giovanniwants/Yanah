# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Validates cache_sim on synthetic routing at Qwen3-Next's dimensions
(48 layers, 512 experts, top-10, 1,600-slot pool):

  A. planted semantic groups: each prompt has topics; a topic owns a group
     of ~25 experts at every layer (different ids per layer); 8 of a
     token's 10 experts come from the topic's group. Topics switch often.
     The lattice SHOULD find the groups and cut demand misses.
  B. no group structure: popularity + temporal reuse only. The lattice
     should NOT help much and should waste most of what it reads.

Plus exact accounting identities. The verdict on the real model comes
only from a real trace (run_mlx_test.py --trace).

    python -m moe_prefetch.test_cache_sim
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from moe_prefetch.cache_sim import (ForecastPolicy, LatticePolicy, Trace, fit_lattice,
                                    load_trace, runs_per_layer, save_trace, seriate, simulate, split)

L, E, K = 48, 512, 10
META = dict(n_layers=L, n_experts=E, top_k=K, slots=1600, spec_slots=64, record_bytes=1769472,
            disk_gbps=5.0)


def forecasts(rng, routed):
    """Router-forecast stand-in: each true target expert with p~0.75, plus decoys."""
    S = len(routed)
    fc = [[{} for _ in range(L)] for _ in range(S)]
    for s in range(S):
        for l in range(L):
            for t in (l + 1, l + 2):
                if t >= L:
                    continue
                c = [(int(e), float(rng.uniform(0.07, 0.3))) for e in routed[s, t] if rng.random() < 0.75]
                c += [(int(e), float(rng.uniform(0.0, 0.1))) for e in rng.integers(0, E, 6)]
                fc[s][l][t] = c
    return fc


def planted(rng, prompts=24, tokens=80, topics=20, switch=0.15):
    groups = []                                       # per layer: topic -> 25 experts
    for _ in range(L):
        perm = rng.permutation(E)
        groups.append([perm[z * 25:(z + 1) * 25] for z in range(topics)])
    routed, prompt = [], []
    for p in range(prompts):
        z = rng.integers(topics)
        for _ in range(tokens):
            if rng.random() < switch:
                z = rng.integers(topics)
            row = []
            for l in range(L):
                ex = list(rng.choice(groups[l][z], 8, replace=False))
                while len(ex) < K:
                    e = int(rng.integers(E))
                    if e not in ex:
                        ex.append(e)
                row.append(ex)
            routed.append(row); prompt.append(p)
    return np.array(routed), np.array(prompt)


def unstructured(rng, prompts=24, tokens=80, reuse=0.5):
    pop = 1.0 / np.arange(1, E + 1) ** 0.8            # Zipf-like popularity, per layer shuffled
    perms = [rng.permutation(E) for _ in range(L)]
    routed, prompt = [], []
    prev = None
    for p in range(prompts):
        for _ in range(tokens):
            row = []
            for l in range(L):
                ex = []
                if prev is not None:
                    ex = [int(e) for e in prev[l] if rng.random() < reuse]
                while len(ex) < K:
                    e = int(perms[l][rng.choice(E, p=pop / pop.sum())])
                    if e not in ex:
                        ex.append(e)
                row.append(ex)
            routed.append(row); prompt.append(p); prev = row
    return np.array(routed), np.array(prompt)


def run(name, routed, prompt, rng):
    t = Trace(routed.astype(np.int64), prompt, forecasts(rng, routed), META)
    train, test = split(t)
    lat = LatticePolicy(fit_lattice(t, train, E, 20, 2, 3))
    f = ForecastPolicy(t)
    res = {}
    for pol, kw, sp in [("cache", {}, 64), ("forecast", dict(forecast=f), 64),
                        ("lattice", dict(lattice=lat), 256), ("both", dict(forecast=f, lattice=lat), 256)]:
        c = simulate(t, test, 1600, sp, **kw)
        assert c["uses"] == c["hits"] + c["adopted"] + c["demand"], c       # every use accounted once
        assert c["prefetch"] == c["adopted"] + c["wasted"], c              # every prefetch used or wasted
        res[pol] = c
    orders = seriate(t, train, E)
    assert all(sorted(o) == list(range(E)) for o in orders), "order must be a permutation"
    pos = [np.argsort(o) for o in orders]
    for key, kw in (("repack", dict(forecast=f)), ("repack_nofc", {})):   # engine as is / cache only
        log = []
        simulate(t, test, 1600, 64, miss_log=log, **kw)
        res[key] = (runs_per_layer(log, None), runs_per_layer(log, pos), float(np.mean([len(x) for _, x in log])))
    print(f"{name}:")
    for pol, c in res.items():
        if pol.startswith("repack"):
            continue
        prec = c["adopted"] / max(1, c["prefetch"])
        print(f"   {pol:9s} demand misses {c['demand_per_tok']:6.1f}/tok | prefetch {c['prefetch_per_tok']:6.1f}/tok "
              f"(precision {prec:5.1%}) | SSD reads {c['reads_per_tok']:6.1f} records/tok")
    return res


def main():
    rng = np.random.default_rng(0)
    A = run("A. planted semantic groups", *planted(rng), rng)
    B = run("B. no group structure", *unstructured(rng), rng)
    cut = lambda r: 1 - r["lattice"]["demand_per_tok"] / r["cache"]["demand_per_tok"]
    prec = lambda r: r["lattice"]["adopted"] / max(1, r["lattice"]["prefetch"])
    print(f"lattice cuts demand misses by {cut(A):.0%} with planted groups vs {cut(B):.0%} without; "
          f"precision {prec(A):.0%} vs {prec(B):.0%}")
    assert cut(A) > 0.3 and cut(A) > 3 * max(cut(B), 0.01), "simulator failed to tell structure from none"
    assert prec(B) < 0.3, "lattice should waste most reads without group structure"
    save = lambda r, k: 1 - r[k][1] / r[k][0]
    for k, label in (("repack_nofc", "cache only"), ("repack", "with router forecast")):
        for nm, r in (("planted groups", A), ("no structure", B)):
            print(f"repacking, {label:20s} ({nm:14s}): {r[k][2]:5.1f} demand misses per layer -> reads "
                  f"{r[k][0]:5.2f} now vs {r[k][1]:5.2f} reordered, {save(r, k):.0%} fewer")
    # the seriation must find planted adjacency when there are many misses per layer ...
    # (oracle that knows the planted groups: 11% fewer reads cache-only, 3% with forecast)
    assert save(A, "repack_nofc") > 0.07 and save(A, "repack_nofc") > 3 * max(save(B, "repack_nofc"), 0.01), \
        "seriation failed to find planted adjacency"
    # ... and with forecasting, little is left to merge even with planted groups
    assert save(A, "repack") < save(A, "repack_nofc")

    # round trip through the on-disk format used by run_mlx_test.py --trace
    entries = []
    pog = {}
    for s in range(3):
        for l in range(L):
            entries.append((100 + s, l, [int(e) for e in A_rt[s, l]], {l + 1: [(1, 0.5), (2, 0.1)]} if l + 1 < L else {}))
        pog[100 + s] = s
    d = tempfile.mkdtemp()
    save_trace(os.path.join(d, "t.npz"), entries, pog, META)
    t2 = load_trace(os.path.join(d, "t.npz"))
    got = t2.fc[1][5][6]                                 # probabilities are stored as float32
    assert np.array_equal(t2.routed, A_rt[:3]) and [e for e, _ in got] == [1, 2]
    assert np.allclose([p for _, p in got], [0.5, 0.1])
    print("trace save/load round trip OK\n\nall cache_sim checks passed")


A_rt = np.random.default_rng(5).integers(0, E, (3, L, K))

if __name__ == "__main__":
    main()
