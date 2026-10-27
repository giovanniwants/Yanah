# SPDX-License-Identifier: FSL-1.1-ALv2
"""
SSD benchmark for the expert file: is one larger contiguous read faster
than the same records read as separate concurrent reads?

That is the whole case for co-activation repacking. Repacking never
reduces BYTES. It can only merge records that sit next to each other on
disk into fewer, larger reads. This measures the best case directly: a
layer's n missing experts read (a) scattered, n concurrent preads, as the
engine does today, vs (b) perfectly adjacent, one pread of n records,
which is the most any reordering could achieve.

Method (as SlotStream's coldread): every record is read AT MOST ONCE,
spread across the whole file, with F_NOCACHE and readahead off, so neither
the page cache nor the SSD's own cache can serve a repeat. F_NOCACHE does
not evict pages that are already cached, so run `sudo purge` first if
experts.bin was written or read recently.

    sudo purge
    python -m moe_prefetch.disk_bench            # default: the 4-bit experts (paths.py)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np

BLOCK = 8          # records per never-reused block (largest contiguous test)


class Bench:
    def __init__(self, expert_dir: str, seed: int = 0):
        spec = json.load(open(os.path.join(expert_dir, "experts.json")))
        self.rec = spec["slot_bytes"]
        self.path = os.path.join(expert_dir, "experts.bin")
        n_rec = os.path.getsize(self.path) // self.rec
        self.fd = os.open(self.path, os.O_RDONLY)
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(self.fd, getattr(fcntl, "F_NOCACHE", 48), 1)
            fcntl.fcntl(self.fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
        self.blocks = list(np.random.default_rng(seed).permutation(n_rec // BLOCK))
        self.pool = cf.ThreadPoolExecutor(32)
        self.bufs = [bytearray(self.rec * BLOCK) for _ in range(32)]
        self.bytes = 0

    def fresh(self) -> int:
        """First record of a never-read block."""
        if not self.blocks:
            raise SystemExit("ran out of unread records; lower --reads/--repeats")
        return int(self.blocks.pop()) * BLOCK

    def _read(self, buf_i: int, record: int, n: int) -> None:
        mv = memoryview(self.bufs[buf_i])[: n * self.rec]
        done, off = 0, record * self.rec
        while done < len(mv):
            k = os.preadv(self.fd, [mv[done:]], off + done)
            if k <= 0:
                raise IOError("short read")
            done += k
        self.bytes += n * self.rec

    def batch(self, reads) -> float:
        """reads: [(record, n_records)] issued concurrently; seconds until all land."""
        t = time.perf_counter()
        fs = [self.pool.submit(self._read, i, r, n) for i, (r, n) in enumerate(reads)]
        for f in fs:
            f.result()
        return time.perf_counter() - t

    def stream(self, size: int, qd: int, reads: int) -> float:
        """GB/s for `reads` reads of `size` records at queue depth qd."""
        t = time.perf_counter()
        pending = set()
        i = 0
        for _ in range(reads):
            if len(pending) >= qd:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            pending.add(self.pool.submit(self._read, i % qd, self.fresh(), size))
            i += 1
        cf.wait(pending)
        return reads * size * self.rec / (time.perf_counter() - t) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from .paths import EXPERTS_4BIT
    ap.add_argument("experts_dir", nargs="?", default=str(EXPERTS_4BIT))
    ap.add_argument("--reads", type=int, default=48, help="reads per throughput cell")
    ap.add_argument("--repeats", type=int, default=25, help="batches per scattered/contiguous cell")
    a = ap.parse_args()
    b = Bench(os.path.expanduser(a.experts_dir))
    print(f"{b.path}: record {b.rec / 2**20:.2f} MiB, {len(b.blocks) * BLOCK} records available, "
          f"F_NOCACHE {'on' if sys.platform == 'darwin' else 'n/a (not macOS: page cache may inflate numbers)'}")

    print("\n[1] throughput (GB/s) by read size and queue depth, never-repeat random offsets")
    sizes, qds = (1, 2, 4), (1, 2, 4, 8, 16)
    print("  read size      " + "".join(f"{'QD' + str(q):>9}" for q in qds))
    for s in sizes:
        row = [b.stream(s, q, max(8, a.reads // s)) for q in qds]
        print(f"  {s} rec {s * b.rec / 2**20:5.2f} MiB " + "".join(f"{v:9.2f}" for v in row))

    print("\n[2] best case for repacking: a layer's n missing experts, scattered vs adjacent")
    print("    (median ms per batch; 'adjacent' = one pread, the most any reordering can give)")
    gains = {}
    for n in (2, 3, 4, 6, 8):
        scat, cont = [], []
        for r in range(a.repeats):
            # alternate order so drift (thermal, background IO) hits both arms equally
            arms = [("s", [(b.fresh(), 1) for _ in range(n)]), ("c", [(b.fresh(), n)])]
            for name, reads in (arms if r % 2 == 0 else arms[::-1]):
                (scat if name == "s" else cont).append(b.batch(reads))
        ms, mc = np.median(scat) * 1e3, np.median(cont) * 1e3
        gains[n] = ms / mc
        print(f"  n={n}: scattered {ms:6.2f} ms ({n} concurrent reads) | adjacent {mc:6.2f} ms (1 read) "
              f"| adjacent is {ms / mc:4.2f}x {'faster' if ms > mc else 'as fast/slower'}")
    print(f"\n  read {b.bytes / 1e9:.2f} GB in total")
    g = np.median(list(gains.values()))
    print(f"  verdict: even PERFECT co-activation adjacency would make a layer's demand reads "
          f"~{g:.2f}x faster on this SSD (median over n). Real reordering only achieves a fraction "
          f"of that: `cache_sim` section [3] measures how many reads it actually merges.")


if __name__ == "__main__":
    main()
