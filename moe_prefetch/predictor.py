# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Router forecast for the next layer (SlotStream's "attention tap").

As each layer's routing comes back, take the residual stream after that
layer's attention / DeltaNet step, apply the NEXT layer's own router to it,
and read out per-expert probabilities. No separately trained predictor:
SlotStream trained small predictors first and stopped them as too
inaccurate; reusing the routers worked. Their measured agreement:
  boundary tap (state two layers back)     0.62
  attention tap (this design)              0.73
  + learned per-layer linear correction    0.80   (35 MiB fp16, optional)
  ceiling (run next attention early)       0.86-0.88, but slower overall

Routers are kept resident in fp32 (SlotStream measured ~2% from avoiding
per-step conversions): 49 x 512 x 2048 x 4 B ~= 205 MiB for Qwen3-Next.
On 8 GB that is a real cost; fp16 halves it.

A learned correction can be plugged in as `correction[layer](logits, h)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .layout import ModelGeometry


def rms_norm(x: np.ndarray, w: Optional[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32)
    y = x32 / np.sqrt(np.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
    return y * w if w is not None else y


@dataclass
class ForecastStats:
    forecasts: int = 0
    candidates: int = 0
    actual: int = 0
    covered: int = 0      # actual experts that appeared among candidates

    @property
    def recall(self) -> float:
        return self.covered / max(1, self.actual)


class RouterForecast:
    """Two modes:

    * backend mode (gate_weights=None): the backend computes each target
      router's probabilities on the GPU from the current post-attention
      state, in the same sync as the routing readback (MLX backend:
      `route_and_forecast`). This class only selects candidates and keeps
      stats. No CPU matmul, no host copy of the routers.
    * numpy mode (gate_weights given): CPU forecast, used by the synthetic
      tests. Always computed in float32: float16 routers made the numpy
      matmul ~24x slower (6.7 ms vs 0.28 ms per layer at Qwen3-Next shapes,
      measured), because numpy has no fast float16 matmul.
    """

    def __init__(self, geometry: ModelGeometry, gate_weights: Optional[Sequence[np.ndarray]] = None,
                 norm_weights: Optional[Sequence[np.ndarray]] = None,
                 correction: Optional[Dict[int, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
                 dtype=np.float32):
        self.g = geometry
        self.gates = ([np.ascontiguousarray(w, dtype=np.float32) for w in gate_weights]
                      if gate_weights is not None else None)
        self.norms = norm_weights
        self.correction = correction or {}
        self._pending: Dict[int, Set[int]] = {}
        self.stats = ForecastStats()

    @property
    def backend_mode(self) -> bool:
        return self.gates is None

    @property
    def resident_bytes(self) -> int:
        return sum(g.nbytes for g in self.gates) if self.gates else 0

    def forecast(self, target_layer: int, h: np.ndarray, top: int = 24) -> List[Tuple[int, float]]:
        """numpy mode: (expert, probability) for the top candidates of
        `target_layer`; for a multi-token pass, an expert's score is its max
        over rows."""
        if self.gates is None or target_layer >= min(self.g.total_moe_layers, len(self.gates)):
            return []
        h = np.atleast_2d(h).astype(np.float32, copy=False)
        w = self.norms[target_layer] if self.norms is not None else None
        x = rms_norm(h, w)
        logits = x @ self.gates[target_layer].T
        if target_layer in self.correction:
            logits = self.correction[target_layer](logits, x)
        z = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)
        return self.select(target_layer, p.max(axis=0), top)

    def select(self, target_layer: int, score: np.ndarray, top: int = 24) -> List[Tuple[int, float]]:
        """Top candidates from per-expert probabilities (E,). Recall is
        tracked over the union of every forecast made for a target, since
        lookahead > 1 forecasts each layer more than once."""
        k = min(top, score.shape[0])
        idx = np.argpartition(-score, k - 1)[:k]
        idx = idx[np.argsort(-score[idx])]
        out = [(int(e), float(score[e])) for e in idx]
        self._pending.setdefault(target_layer, set()).update(e for e, _ in out)
        self.stats.forecasts += 1
        self.stats.candidates += len(out)
        return out

    def observe(self, layer: int, routed: np.ndarray) -> None:
        actual = set(int(e) for e in np.unique(routed))
        pred = self._pending.pop(layer, None)
        if pred is not None:
            self.stats.actual += len(actual)
            self.stats.covered += len(actual & pred)

    def reset(self) -> None:
        self._pending.clear()
