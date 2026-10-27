# SPDX-License-Identifier: FSL-1.1-ALv2
"""
MTP self-speculative decoding with state-recording rollback and
SlotStream's wall-clock adaptive depth (AdaptiveSpeculation.swift, MIT).

Round:
  1. Draft d tokens with the MTP head (chained).
  2. Verify [last, d1..dd] in ONE pool-path forward with recording on.
  3. Greedy: accept the longest matching prefix + the main model's next
     token. Sampled: standard rejection sampling.
  4. backend.rollback(n+1): recurrent states <- recorded position n, KV
     trimmed. SlotStream measured re-running the kept tokens as the single
     largest avoidable cost in speculation; recording removes it.

Adaptive depth (ported policy): calibrate on 3 plain single-token steps,
then draft. Compare the median cost per EMITTED token of the last 3 draft
rounds with the median plain step. If drafting is >5% slower, halve the
depth; at depth 1, stop drafting for the request. Recalibrate every 16
rounds or 256 tokens of context growth.

8 GB warning, from SlotStream's measurements (mtp-bench, 4 drafts):
x0.55 / 0.69 / 0.88 / 0.96 at 20 / 29 / 42 / 57 experts per layer, all
BELOW break-even. On a fetch-bound pool, every rejected draft still read
its experts from SSD. They enable MTP only from ~76 experts/layer. An 8 GB
plan lands near 25-30/layer, so expect this policy to switch MTP off; use
`mtp_allowed()` from plan.py as the gate and K=3 only as the starting depth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .orchestrator import MoEOrchestrator


class AdaptiveSpeculationPolicy:
    CALIBRATE, PLAIN = "calibrate", "plain"

    def __init__(self, maximum_depth: int):
        self.depth = min(16, max(1, maximum_depth))
        self.plain: List[float] = []
        self.draft: List[float] = []
        self.stopped = False
        self.baseline_ctx = 0
        self.rounds_since = 0

    def action(self, ctx: int):
        if self.stopped:
            return self.PLAIN
        if len(self.plain) == 3 and (self.rounds_since >= 16 or ctx - self.baseline_ctx >= 256):
            self.plain.clear(); self.draft.clear(); self.rounds_since = 0
        return self.CALIBRATE if len(self.plain) < 3 else self.depth

    def observe_plain(self, seconds: float, ctx: int) -> None:
        if len(self.plain) == 3:
            self.plain.pop(0)
        self.plain.append(seconds)
        self.baseline_ctx = ctx

    def observe_draft(self, seconds: float, emitted: int) -> None:
        self.rounds_since += 1
        self.draft.append(seconds / emitted)
        if len(self.draft) > 3:
            self.draft.pop(0)
        if len(self.draft) < 3 or len(self.plain) < 3:
            return
        ref, cand = sorted(self.plain)[1], sorted(self.draft)[1]
        if cand > ref * 1.05:
            if self.depth == 1:
                self.stopped = True
            else:
                self.depth = max(1, self.depth // 2)
                self.draft.clear()


@dataclass
class SpecStats:
    rounds: int = 0
    plain_steps: int = 0
    drafted: int = 0
    accepted: int = 0
    emitted: int = 0
    depth_hist: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"draft rounds={self.rounds} plain steps={self.plain_steps} "
                f"acceptance={self.accepted / max(1, self.drafted):.1%} depths={self.depth_hist}")


def _softmax(x, temp):
    z = (x - x.max()) / temp
    p = np.exp(z)
    return p / p.sum()


class MTPSpeculator:
    def __init__(self, orch: MoEOrchestrator, max_depth: int = 3, adaptive: bool = True,
                 enabled: bool = True, temperature: float = 0.0, prefill_chunk: int = 256,
                 seed: int = 0):
        self.o, self.b = orch, orch.b
        self.policy = AdaptiveSpeculationPolicy(max_depth) if adaptive else None
        self.fixed = max_depth
        self.enabled = enabled
        # Forecast the MTP layer's experts only while drafting; otherwise those
        # reads are wasted bandwidth.
        orch.forecast_mtp = bool(enabled and getattr(orch.b, "has_mtp", True))
        self.temp = temperature
        self.chunk = prefill_chunk
        self.rng = np.random.default_rng(seed)
        self.stats = SpecStats()

    def _pick(self, logits):
        if self.temp <= 0:
            return int(np.argmax(logits)), None
        p = _softmax(logits, self.temp)
        return int(self.rng.choice(len(p), p=p)), p

    def generate(self, prompt: List[int], max_new: int, eos: Optional[int] = None) -> List[int]:
        if self.o.fc is not None:
            self.o.fc.reset()
        logits, hidden = self.o.prefill(prompt, self.chunk)
        tok, _ = self._pick(logits[-1])
        out, h_last, ctx = [tok], hidden[-1], len(prompt)
        # Keep the MTP head's own attention cache aligned with committed
        # positions (true hidden states), when the backend has one.
        sync = getattr(self.b, "mtp_sync", None) if self.enabled else None
        if sync:
            sync(hidden, list(prompt[1:]) + [tok])

        while len(out) < max_new and tok != eos:
            action = self.policy.action(ctx) if (self.policy and self.enabled) else None
            if not self.enabled:
                d = 0
            elif self.policy is None:
                d = self.fixed
            else:
                d = 0 if action in (AdaptiveSpeculationPolicy.PLAIN, AdaptiveSpeculationPolicy.CALIBRATE) else action
            d = min(d, max_new - len(out))
            t0 = time.perf_counter()
            if d == 0:                                           # plain step
                logits, hidden = self.o.forward([tok])
                nxt, _ = self._pick(logits[0])
                new, h_last = [nxt], hidden[0]
                if sync:
                    sync(hidden[0:1], [nxt])
                if action == AdaptiveSpeculationPolicy.CALIBRATE:
                    self.policy.observe_plain(time.perf_counter() - t0, ctx)
                self.stats.plain_steps += 1
            else:
                drafts, qs = [], []
                dh, dt = h_last, tok
                for _ in range(d):
                    dl, dh = self.o.mtp_forward(dh, dt)
                    dt, q = self._pick(dl)
                    drafts.append(dt); qs.append(q)
                self.b.mtp_reset()
                logits, hidden = self.o.forward([tok] + drafts, record=True)
                n, nxt = 0, None
                for i, dtok in enumerate(drafts):
                    if self.temp <= 0:
                        ok = int(np.argmax(logits[i])) == dtok
                    else:
                        p = _softmax(logits[i], self.temp)
                        ok = self.rng.random() < min(1.0, p[dtok] / max(qs[i][dtok], 1e-12))
                        if not ok:
                            r = np.maximum(p - qs[i], 0)
                            nxt = int(self.rng.choice(len(r), p=r / r.sum())) if r.sum() > 0 else self._pick(logits[i])[0]
                    if not ok:
                        break
                    n += 1
                if nxt is None:
                    nxt, _ = self._pick(logits[n])
                self.b.rollback(n + 1)                           # no recompute
                new, h_last = drafts[:n] + [nxt], hidden[n]
                if sync:
                    sync(hidden[:n + 1], new)
                if self.policy:
                    self.policy.observe_draft(time.perf_counter() - t0, len(new))
                self.stats.rounds += 1
                self.stats.drafted += d
                self.stats.accepted += n
                self.stats.depth_hist[d] = self.stats.depth_hist.get(d, 0) + 1
            out.extend(new)
            ctx += len(new)
            tok = nxt
            self.stats.emitted += len(new)
            if eos is not None and eos in new:
                out = out[:out.index(eos) + 1]
                break
        return out[:max_new]
