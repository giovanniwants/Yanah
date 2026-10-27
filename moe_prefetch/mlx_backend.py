# SPDX-License-Identifier: FSL-1.1-ALv2
"""
MLX backend for Qwen3-Next (mlx-lm's qwen3_next module), plugging into
orchestrator.ModelBackend.

Memory model
------------
* Dense weights (embeddings, attention, Gated DeltaNet, norms, routers,
  shared experts, lm_head) are loaded from the checkpoint and stay resident.
  The stacked `switch_mlp` expert tensors are never evaluated: the model is
  built with `load_model(lazy=True)`, the switch_mlp modules are removed,
  and only what remains is materialised.
* Routed experts live in `MLXExpertMemory`: one uint8 MLX array per weight
  piece (gate/up/down x weight/scales/biases), each (slots, piece_bytes),
  with typed zero-copy views (slots, *shape). The IO threads preadv straight
  into these buffers through the buffer protocol, and `mx.gather_qmm`
  indexes the pool rows directly, so an expert is never copied after the
  read. Apple Silicon memory is unified, so the GPU reads the same pages.
* Ordering contract: `moe()` evaluates its output before returning, because
  the orchestrator unpins the slots right after and a later demand read
  may overwrite them. Router logits must reach the CPU anyway (routing,
  forecasting), so each layer already has a sync point.

Gated DeltaNet rollback
-----------------------
mlx-lm's GatedDeltaNet keeps two pieces of state per layer: the conv1d
window (last kernel-1 inputs) and the recurrent state (B, Hv, Dv, Dk) fp32.
Neither can be rewound. During a recorded (speculative verify) pass this
backend steps `gated_delta_update` one token at a time and keeps both after
every position; `rollback(keep)` swaps in position keep-1 and trims the
full-attention KV caches. No recompute. Cost: (T x 36 layers x ~2 MiB) of
state during a T-token verify pass, ~290 MiB at depth 3.

Exactness
---------
On the pool path (decode, verify, short prompts) this reproduces mlx-lm's
Qwen3NextSparseMoeBlock op for op (same softmax/argpartition routing, same
SwitchGLU sort rule, routed + shared summed before the residual), so logits
match the stock model. The prefill sweep sums per staging group and is not
bit-identical (same as SlotStream's sweep).

MTP
---
Standard mlx-lm conversions DROP the MTP weights (`sanitize` removes mtp.*).
The MTP head here loads only if the checkpoint keeps them (mtp.fc,
mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden, mtp.layers.0.*, mtp.norm,
experts repacked as layer n_layers). Two details to verify against a
reference implementation before trusting acceptance numbers: whether the
hidden fed to MTP is post final norm (`mtp_hidden_post_norm`, vLLM's
convention) and whether the checkpoint's norm weights still need mlx-lm's
+1.0 (`mtp_norm_offset`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.models.qwen3_next import ModelArgs, Qwen3NextDecoderLayer, Qwen3NextSparseMoeBlock
from mlx_lm.utils import infer_quant_config, load_model

from .layout import ExpertLayout, ModelGeometry
from .repack import RecordSpec, read_headers

_MX_DTYPE = {"U32": mx.uint32, "U8": mx.uint8, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
_NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight",
                  ".k_norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")


# ------------------------------------------------------------ slot memory
class MLXExpertMemory:
    """Pool or staging memory: per-piece MLX arrays that IO writes into."""

    def __init__(self, n: int, spec: RecordSpec):
        self.n, self.spec = n, spec
        self.raw: List[mx.array] = []
        self.arrays: Dict[str, mx.array] = {}
        for p in spec.pieces:
            raw = mx.zeros((n, p.nbytes), dtype=mx.uint8)
            typed = raw.view(_MX_DTYPE[p.dtype]).reshape(n, *p.shape)
            mx.eval(raw, typed)
            self.raw.append(raw)
            self.arrays[p.name] = typed
        self._flat = [memoryview(r).cast("B") for r in self.raw]
        self._nb = [p.nbytes for p in spec.pieces]
        for f, nb in zip(self._flat, self._nb):
            if f.readonly or f.nbytes != n * nb:
                raise RuntimeError("MLX array is not a writable dense buffer; cannot stream into it")

    def views(self, i: int) -> List[memoryview]:
        return [f[i * nb:(i + 1) * nb] for f, nb in zip(self._flat, self._nb)]

    @property
    def nbytes(self) -> int:
        return self.n * sum(self._nb)

    def close(self) -> None:
        for f in self._flat:
            f.release()
        self._flat, self.raw, self.arrays = [], [], {}


def _gather_sort(x, indices):            # mlx_lm.models.switch_layers._gather_sort
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


# ---------------------------------------------------------------- MTP head
class MTPHead(nn.Module):
    """Qwen3-Next multi-token-prediction block: fuse(norm(embed(t+1)),
    norm(h_t)) -> one full-attention MoE decoder layer -> norm -> lm_head."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        H = args.hidden_size
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.layers = [Qwen3NextDecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(H, eps=args.rms_norm_eps)


# ----------------------------------------------------------------- backend
class MLXQwen3NextBackend:
    def __init__(self, model_dir: str, expert_dir: str, cache_limit_gb: float = 0.5,
                 load_mtp: bool = True, mtp_hidden_post_norm: bool = True,
                 mtp_norm_offset: float = 1.0, kv_bits: Optional[int] = None,
                 kv_group_size: int = 64):
        """kv_bits: quantize the KV cache of the full-attention layers (12 of
        48 in Qwen3-Next) with mlx-lm's QuantizedKVCache. 24 KiB -> 6.75 KiB
        per context token at 4 bits (measured). The Gated DeltaNet state is
        fixed-size and untouched. Can be changed between requests: set
        `kv_bits` and call reset()."""
        self.kv_bits, self.kv_group_size = kv_bits, kv_group_size
        try:
            mx.set_cache_limit(int(cache_limit_gb * 2**30))
        except AttributeError:
            mx.metal.set_cache_limit(int(cache_limit_gb * 2**30))

        with open(os.path.join(expert_dir, "experts.json")) as f:
            self.spec = RecordSpec.from_json(json.load(f))
        self.expert_file = os.path.join(expert_dir, "experts.bin")

        # Lazy load: arrays are file-backed until evaluated, so the 40+ GB of
        # stacked experts are never read. strict=False tolerates mtp.* keys;
        # coverage of every dense parameter is checked explicitly below.
        model, self.config = load_model(Path(model_dir), lazy=True, strict=False)
        self.model, self.args = model, model.args
        for i, L in enumerate(model.layers):
            if not isinstance(L.mlp, Qwen3NextSparseMoeBlock):
                raise NotImplementedError(f"layer {i} is dense; this backend assumes every layer is MoE")
            L.mlp.pop("switch_mlp")                    # experts stream from the pool
        headers = read_headers(model_dir)
        missing = [k for k, _ in _flatten(model.parameters()) if k not in headers]
        if missing:
            raise ValueError(f"checkpoint lacks dense weights, e.g. {missing[:5]}")
        mx.eval(model.parameters())

        self.mtp: Optional[MTPHead] = None
        self.mtp_post_norm = mtp_hidden_post_norm
        if load_mtp and any(k.startswith("mtp.fc.") for k in headers):
            self.mtp = self._load_mtp(model_dir, headers, mtp_norm_offset)

        s = self.spec
        gate = {p.name: p for p in s.pieces}
        H = self.args.hidden_size
        self.bits = gate["gate_proj.weight"].shape[-1] * 32 // H
        self.group_size = H // gate["gate_proj.scales"].shape[-1]
        self.geometry = ModelGeometry(
            n_layers=self.args.num_hidden_layers, n_mtp_layers=s.n_mtp_layers,
            n_experts=self.args.num_experts, top_k=self.args.num_experts_per_tok,
            hidden=H, moe_inter=self.args.moe_intermediate_size, bits=self.bits,
            group_size=self.group_size, record_bytes=s.record_bytes)
        if s.n_layers != self.args.num_hidden_layers or s.n_experts != self.args.num_experts:
            raise ValueError("experts.json does not match config.json")

        self.cache = self._make_cache()
        self.mtp_cache = KVCache()
        self.recording = False
        self._rec: Dict[int, list] = {}
        self._attn_T: Dict[int, int] = {}
        self._shared: Dict[int, mx.array] = {}
        self._mtp_pending = 0
        self._unfenced: Optional[mx.array] = None   # submitted layer output not yet known complete
        self.batch_mode = False                      # rows are B sequences (one token each)

    # ---------------------------------------------------------------- setup
    def _make_cache(self) -> list:
        cache = self.model.make_cache()
        if self.kv_bits:
            cache = [QuantizedKVCache(self.kv_group_size, self.kv_bits) if isinstance(c, KVCache) else c
                     for c in cache]
        return cache

    def snapshot(self):
        """A restorable point in the sequence: references to every Gated
        DeltaNet layer's conv/recurrent arrays (MLX arrays are immutable, so
        this copies nothing; it keeps ~74 MiB alive) and each attention
        cache's length (KV caches can be trimmed back to it)."""
        self.fence()
        return [("arrays", c[0], c[1]) if not isinstance(c, (KVCache, QuantizedKVCache))
                else ("kv", c.offset) for c in self.cache]

    def restore(self, snap) -> None:
        """Rewind the state to a snapshot taken earlier in THIS sequence."""
        self.fence()
        for c, s in zip(self.cache, snap):
            if s[0] == "arrays":
                c[0], c[1] = s[1], s[2]
            else:
                if c.offset < s[1]:
                    raise ValueError("snapshot is ahead of the cache")
                c.trim(c.offset - s[1])

    def kv_nbytes(self) -> int:
        """Bytes currently held by the attention KV caches (allocated, in
        256-token steps), excluding the fixed Gated DeltaNet state."""
        return sum(c.nbytes for c in self.cache
                   if isinstance(c, (KVCache, QuantizedKVCache)) and not c.empty())

    def layout(self) -> ExpertLayout:
        return ExpertLayout.contiguous(self.geometry, self.expert_file)

    def make_expert_memory(self, n: int) -> MLXExpertMemory:
        return MLXExpertMemory(n, self.spec)

    def _load_mtp(self, model_dir, headers, norm_offset) -> MTPHead:
        head = MTPHead(self.args)
        head.layers[0].mlp.pop("switch_mlp")
        w: Dict[str, mx.array] = {}
        for f in sorted({headers[k][0] for k in headers if k.startswith("mtp.")}):
            for k, v in mx.load(f).items():
                if k.startswith("mtp.") and ".switch_mlp." not in k:
                    w[k[4:]] = v
        for k in list(w):
            if (k.endswith(_NORM_SUFFIXES) or k == "norm.weight") and w[k].ndim == 1 and norm_offset:
                w[k] = w[k] + norm_offset

        def pred(p, m):
            if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
                return False
            return infer_quant_config(p, m, w)
        nn.quantize(head, class_predicate=pred)
        head.load_weights(list(w.items()), strict=True)
        mx.eval(head.parameters())
        return head

    def forecast_weights(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Dequantised routers (fp16) and the norms that feed them, for RouterForecast."""
        layers = list(self.model.layers) + ([self.mtp.layers[0]] if self.mtp else [])
        gates, norms = [], []
        for L in layers:
            g = L.mlp.gate
            W = (mx.dequantize(g.weight, g.scales, g.biases, g.group_size, g.bits)
                 if isinstance(g, nn.QuantizedLinear) else g.weight)
            gates.append(np.array(W.astype(mx.float16)))
            norms.append(np.array(L.post_attention_layernorm.weight.astype(mx.float32)))
        return gates, norms

    # ----------------------------------------------------- ModelBackend API
    def embed(self, tokens) -> mx.array:
        return self.model.model.embed_tokens(mx.array(np.asarray(tokens), dtype=mx.int32))

    def _to_batch(self, h: mx.array) -> mx.array:
        """Rows -> mlx-lm's (B, L, H): one sequence of T tokens, or (batch
        mode) B sequences of one token each."""
        return h[:, None, :] if self.batch_mode else h[None]

    def _to_rows(self, a: mx.array) -> mx.array:
        return a[:, 0] if self.batch_mode else a[0]

    def pre_moe(self, layer: int, h: mx.array):
        L, c = self.model.layers[layer], self.cache[layer]
        hb = self._to_batch(h)
        xin = L.input_layernorm(hb)
        if L.is_linear:
            r = self._gated_delta(L.linear_attn, xin, c, layer)
        else:
            r = L.self_attn(xin, create_attention_mask(hb, c), c)
            if self.recording:
                self._attn_T[layer] = hb.shape[1]
        return self._post_attention(L, hb + r, layer)

    def _post_attention(self, L, h1: mx.array, layer: int):
        x = L.post_attention_layernorm(h1)
        m = L.mlp
        self._shared[layer] = self._to_rows(mx.sigmoid(m.shared_expert_gate(x)) * m.shared_expert(x))
        return self._to_rows(h1), self._to_rows(x), self._to_rows(m.gate(x))

    def route(self, logits: mx.array):
        """Exactly Qwen3NextSparseMoeBlock's routing."""
        gates = mx.softmax(logits, axis=-1, precise=True)
        k = self.args.num_experts_per_tok
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.args.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return np.array(inds), scores          # np.array syncs: attention + router done

    def _router_layer(self, t: int):
        n = self.args.num_hidden_layers
        if t < n:
            return self.model.layers[t]
        if t == n and self.mtp is not None:
            return self.mtp.layers[0]
        return None

    def route_and_forecast(self, logits: mx.array, h_res: mx.array, targets: Sequence[int]):
        """route() plus each target layer's own router (the resident 8-bit
        `mlp.gate`) applied to this layer's post-attention residual, after
        the target's own post-attention norm: SlotStream's attention tap.
        Everything is materialised in ONE sync, the one routing needs
        anyway, so forecasting adds no CPU matmul and no extra GPU wait.
        Returns (ids, scores, {target: per-expert probability (E,)})."""
        gates = mx.softmax(logits, axis=-1, precise=True)
        k = self.args.num_experts_per_tok
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.args.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        fc = {}
        for t in targets:
            L = self._router_layer(t)
            if L is None:
                continue
            p = mx.softmax(L.mlp.gate(L.post_attention_layernorm(h_res)).astype(mx.float32),
                           axis=-1, precise=True)
            fc[t] = p.max(axis=0)              # (E,): max over rows of a multi-token pass
        mx.eval(inds, *fc.values())
        return np.array(inds), scores, {t: np.array(p) for t, p in fc.items()}

    # ------------------------------------------- lazy experts + write fence
    # The pool path no longer blocks on the experts (`sync=False`): layer l's
    # expert graph is submitted with async_eval and runs while Python builds
    # layer l+1, then completes inside layer l+1's routing sync. That is one
    # GPU round trip per layer instead of two.
    #
    # Safety invariant: no pool slot may be written while a submitted graph
    # that reads it may still be pending. Every pool write decision (demand
    # victims, partition refills, forecast reservations) is made by the
    # orchestrator after `fence()`. The fence is free in practice, because
    # the routing sync before it already evaluated the previous layer's
    # output. The sweep keeps `sync=True`: its double-buffered staging is
    # refilled by IO threads while a lazy group would still be pending.
    supports_lazy_moe = True

    def launch(self, h: mx.array) -> None:
        """Start executing a layer's output without waiting for it."""
        mx.async_eval(h)
        self._unfenced = h

    def fence(self) -> None:
        """Block until every submitted graph has finished reading pool slots."""
        if self._unfenced is not None:
            mx.eval(self._unfenced)
            self._unfenced = None

    def moe_ready(self, layer: int, x: mx.array, ids: np.ndarray, memory: MLXExpertMemory,
                  ready: Dict[int, int]):
        """Decode (T == 1): build the routed outputs of the experts already in
        memory and start them on the GPU, together with the shared expert,
        BEFORE the orchestrator blocks on the misses. Same per-expert SwitchGLU
        call as the stock path, just over a subset of the k columns, so
        `moe_rest` can reassemble the exact stock (1, k, H) tensor."""
        cols = [j for j in range(ids.shape[1]) if int(ids[0, j]) in ready]
        y = None
        if cols:
            idx = mx.array([[ready[int(ids[0, j])] for j in cols]], dtype=mx.int32)
            y = self._switch_glu(x, idx, memory.arrays)               # (1, kr, H)
            mx.async_eval(y, self._shared[layer])
        else:
            mx.async_eval(self._shared[layer])
        return cols, y

    def moe_rest(self, layer: int, x: mx.array, ids: np.ndarray, w: mx.array,
                 memory: MLXExpertMemory, rows: Dict[int, int], handle) -> mx.array:
        """Remaining columns, then the stock weighted sum over the k columns
        in their original order. Lazy: no sync."""
        cols, y_ready = handle
        k = ids.shape[1]
        rest = [j for j in range(k) if j not in set(cols)]
        parts, order = [], []
        if y_ready is not None:
            parts.append(y_ready); order += cols
        if rest:
            idx = mx.array([[rows[int(ids[0, j])] for j in rest]], dtype=mx.int32)
            parts.append(self._switch_glu(x, idx, memory.arrays)); order += rest
        y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        if order != list(range(k)):
            y = y[:, mx.array(np.argsort(order).astype(np.int32))]      # exact permutation
        return (y * w[..., None]).sum(axis=-2)

    def moe(self, layer: int, x: mx.array, ids: np.ndarray, w: mx.array,
            memory: MLXExpertMemory, rows: Dict[int, int], sync: bool = True) -> mx.array:
        A = memory.arrays
        lut = np.zeros(self.args.num_experts, np.int32)
        lut[list(rows.keys())] = list(rows.values())
        present = np.isin(ids, np.fromiter(rows.keys(), dtype=np.int64))
        if present.all():
            # pool path: identical to SwitchGLU, rhs indices are pool rows
            y = self._switch_glu(x, mx.array(lut[ids]), A)            # (T, k, H)
            out = (y * w[..., None]).sum(axis=-2)
        else:
            # sweep path: only this staging group's assignments, rows sorted
            t_idx, j_idx = np.nonzero(present)
            r = lut[ids[t_idx, j_idx]]
            order = np.argsort(r, kind="stable")
            t_idx, j_idx, r = t_idx[order], j_idx[order], r[order]
            xs = x[mx.array(t_idx)][:, None, :]                       # (n, 1, H)
            idx = mx.array(r)
            up = self._qmm(xs, A, "up_proj", idx, True)
            gate = self._qmm(xs, A, "gate_proj", idx, True)
            y = self._qmm(swiglu(gate, up), A, "down_proj", idx, True)[:, 0, :]
            ws = mx.take(w.reshape(-1), mx.array(t_idx * ids.shape[1] + j_idx))
            out = mx.zeros_like(x).at[mx.array(t_idx)].add(y * ws[:, None])
        if sync:
            mx.eval(out)    # slots may be overwritten as soon as we return
        return out

    def moe_finalize(self, layer: int, routed: mx.array) -> mx.array:
        """routed + shared, in the stock block's order, before the residual."""
        return routed + self._shared.pop(layer)

    def _qmm(self, x, A, proj, idx, sorted_indices):
        return mx.gather_qmm(x, A[f"{proj}.weight"], A[f"{proj}.scales"], A.get(f"{proj}.biases"),
                             rhs_indices=idx, transpose=True, group_size=self.group_size,
                             bits=self.bits, mode="affine", sorted_indices=sorted_indices)

    def _switch_glu(self, x, inds, A):
        """mlx_lm SwitchGLU.__call__ on (T, H) with pool-row indices (T, k)."""
        x = mx.expand_dims(x, (-2, -3))
        do_sort = inds.size >= 64
        idx, inv = inds, None
        if do_sort:
            x, idx, inv = _gather_sort(x, inds)
        up = self._qmm(x, A, "up_proj", idx, do_sort)
        gate = self._qmm(x, A, "gate_proj", idx, do_sort)
        y = self._qmm(swiglu(gate, up), A, "down_proj", idx, do_sort)
        if do_sort:
            y = _scatter_unsort(y, inv, inds.shape)
        return y.squeeze(-2)

    def lm_logits(self, h: mx.array) -> np.ndarray:
        out = self.model.model.norm(self._to_batch(h))
        out = (self.model.model.embed_tokens.as_linear(out) if self.args.tie_word_embeddings
               else self.model.lm_head(out))
        return np.array(self._to_rows(out).astype(mx.float32))

    # ------------------------------------------------------- batched decode
    # Mirrors mlx-lm's BatchGenerator: each sequence is prefilled into its
    # own cache, the caches are merged with each cache type's own `merge`
    # (left-padded BatchKVCache for attention, stacked conv/recurrent state
    # for Gated DeltaNet), every decode step is one pass over B rows, and
    # finished sequences are dropped with `filter`. In the expert path, B
    # rows are just a pass of B rows: each layer reads the UNION of their
    # experts once and every row uses it.
    def take_cache(self) -> list:
        """Detach the current single-sequence cache (after its prefill)."""
        self.fence()
        c, self.cache = self.cache, self._make_cache()
        return c

    def begin_batch(self, caches: Sequence[list]) -> None:
        if self.kv_bits:
            raise NotImplementedError("mlx-lm's QuantizedKVCache cannot be merged into a batch")
        if self.recording:
            raise RuntimeError("speculative recording is not supported in batch mode")
        self.fence()
        self.cache = [caches[0][i].merge([c[i] for c in caches]) for i in range(len(caches[0]))]
        self.batch_mode = True

    def filter_batch(self, keep: Sequence[int]) -> None:
        """Keep only these rows (finished sequences leave the batch)."""
        self.fence()
        for c in self.cache:
            c.filter(list(keep))

    def end_batch(self) -> None:
        self.fence()
        self.batch_mode = False
        self.cache = self._make_cache()

    def forecast_input(self, h: mx.array) -> np.ndarray:
        return np.array(h.astype(mx.float32))

    # ------------------------------------------- Gated DeltaNet (+recording)
    def _gated_delta(self, m, x: mx.array, cache, layer: int) -> mx.array:
        """mlx-lm Qwen3NextGatedDeltaNet.__call__ for batch 1, plus per-position
        state recording when self.recording is set."""
        mask = create_ssm_mask(x, cache)       # None unless the batch cache carries padding
        if mask is not None and self.recording:
            raise NotImplementedError("speculative recording with a padded batch")
        B, S, _ = x.shape
        q, k, v, z, b, a = m.fix_query_key_value_ordering(m.in_proj_qkvz(x), m.in_proj_ba(x))
        conv_state = cache[0] if cache[0] is not None else mx.zeros(
            (B, m.conv_kernel_size - 1, m.conv_dim), dtype=x.dtype)
        mixed = mx.concatenate([q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1)
        if mask is not None:                   # as mlx-lm's Qwen3NextGatedDeltaNet
            mixed = mx.where(mask[..., None], mixed, 0)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        n_keep = m.conv_kernel_size - 1
        if getattr(cache, "lengths", None) is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(m.conv1d(conv_input))
        q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
            mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1),
            [m.num_k_heads, m.num_k_heads, m.num_v_heads],
            [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
        q, k = normalize_qk(q, k, inv_scale=m.head_k_dim ** -0.5, eps=1e-6)
        state = cache[1]
        if self.recording:
            rec, outs = self._rec.setdefault(layer, []), []
            for i in range(S):
                o, state = gated_delta_update(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1],
                                              a[:, i:i + 1], b[:, i:i + 1], m.A_log, m.dt_bias,
                                              state, None, use_kernel=True)
                outs.append(o)
                rec.append((mx.contiguous(conv_input[:, i + 1:i + 1 + n_keep, :]), state))
            out = mx.concatenate(outs, axis=1)
        else:
            out, state = gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, state, mask,
                                            use_kernel=True)
        cache[1] = state
        cache.advance(S)
        out = m.norm(out, z)
        return m.out_proj(out.reshape(B, S, -1))

    def set_recording(self, on: bool) -> None:
        self.recording = on
        self._rec, self._attn_T = {}, {}

    def rollback(self, keep: int) -> None:
        """Keep the first `keep` positions of the recorded pass. No compute."""
        for layer, rec in self._rec.items():
            if keep < len(rec):
                conv, state = rec[keep - 1]
                self.cache[layer][0], self.cache[layer][1] = conv, state
        for layer, T in self._attn_T.items():
            self.cache[layer].trim(T - keep)
        self.set_recording(False)

    # ----------------------------------------------------------------- MTP
    @property
    def has_mtp(self) -> bool:
        return self.mtp is not None

    def _mtp_inputs(self, h_prev: mx.array, tokens) -> mx.array:
        hp = h_prev[None]
        if self.mtp_post_norm:
            hp = self.model.model.norm(hp)
        e = self.model.model.embed_tokens(mx.array([list(tokens)], dtype=mx.int32))
        return self.mtp.fc(mx.concatenate(
            [self.mtp.pre_fc_norm_embedding(e), self.mtp.pre_fc_norm_hidden(hp)], axis=-1))

    def mtp_pre_moe(self, h_prev: mx.array, next_token: int):
        if self.mtp is None:
            raise RuntimeError("checkpoint has no MTP weights; run with speculation disabled")
        z = self._mtp_inputs(h_prev, [next_token])
        L = self.mtp.layers[0]
        r = L.self_attn(L.input_layernorm(z), create_attention_mask(z, self.mtp_cache), self.mtp_cache)
        self._mtp_pending += 1
        return self._post_attention(L, z + r, self.geometry.n_layers)

    def mtp_logits(self, h: mx.array) -> np.ndarray:
        out = self.mtp.norm(h[None])
        out = (self.model.model.embed_tokens.as_linear(out) if self.args.tie_word_embeddings
               else self.model.lm_head(out))
        return np.array(out[0].astype(mx.float32))

    def mtp_reset(self) -> None:
        """Drop the draft chain's MTP KV entries."""
        if self._mtp_pending:
            self.mtp_cache.trim(self._mtp_pending)
            self._mtp_pending = 0

    def mtp_sync(self, hiddens: mx.array, next_tokens: Sequence[int]) -> None:
        """Append MTP KV for committed positions (true hidden states). Only the
        attention K/V are needed, so the MTP layer's MoE is skipped."""
        if self.mtp is None or len(next_tokens) == 0:
            return
        z = self._mtp_inputs(hiddens, next_tokens)
        L = self.mtp.layers[0]
        L.self_attn(L.input_layernorm(z), create_attention_mask(z, self.mtp_cache), self.mtp_cache)
        mx.eval(self.mtp_cache.keys, self.mtp_cache.values)

    # --------------------------------------------------------- diagnostics
    def check_stepping(self, layer: int, T: int = 4, seed: int = 0) -> float:
        """Max |fused - stepped| for one GDN layer's output and state.
        Run once on the target Mac; it should be 0 (or bf16-ulp small)."""
        m = self.model.layers[layer].linear_attn
        x = mx.random.normal((1, T, self.args.hidden_size), key=mx.random.key(seed)).astype(
            self.model.model.norm.weight.dtype)
        from mlx_lm.models.cache import ArraysCache
        c1, c2 = ArraysCache(2), ArraysCache(2)
        saved = self.recording
        self.recording = False
        y1 = self._gated_delta(m, x, c1, layer)
        self.recording, self._rec = True, {}
        y2 = self._gated_delta(m, x, c2, layer)
        self.recording, self._rec = saved, {}
        d = max(mx.abs(y1 - y2).max().item(), mx.abs(c1[1] - c2[1]).max().item())
        return float(d)

    def reset(self) -> None:
        self.fence()
        self.batch_mode = False
        self.cache = self._make_cache()
        self.mtp_cache = KVCache()
        self.set_recording(False)
        self._shared.clear()
        self._mtp_pending = 0


def _flatten(tree):
    from mlx.utils import tree_flatten
    return tree_flatten(tree)


# --------------------------------------------------------------- wiring
def build_runtime(model_dir: str, expert_dir: str, process_target_gb: float = 5.5,
                  mtp_depth: int = 0, use_forecast: bool = True, sweep_min_tokens: int = 16,
                  spec_lanes: int = 8, spec_slot_cap: int = 64, cache_limit_gb: float = 0.5,
                  slots: Optional[int] = None, lookahead: int = 2, lazy_moe: bool = True,
                  overlap_resident: bool = True, **backend_kw):
    """Backend + pool + IO + forecasts + sweep + orchestrator for one process.
    `slots` overrides the plan (use the plan's number only after replacing
    its estimates with measurements)."""
    from .io_lanes import ExpertReader, IOLaneBudget, SpeculativeScheduler
    from .orchestrator import MoEOrchestrator
    from .plan import MemoryPlan
    from .pool import SlotPool
    from .predictor import RouterForecast
    from .sweep import PrefillSweep

    b = MLXQwen3NextBackend(model_dir, expert_dir, cache_limit_gb=cache_limit_gb, **backend_kw)
    g = b.geometry
    if slots is None:
        plan = MemoryPlan(process_target_gb=process_target_gb, mtp_depth=mtp_depth,
                          spec_slot_cap=spec_slot_cap, mlx_cache_cap_gb=cache_limit_gb, g=g)
        plan.validate()
        slots = plan.slots
    pool = SlotPool(g, slots, spec_slot_cap=spec_slot_cap, memory=b.make_expert_memory(slots))
    reader = ExpertReader(b.layout())
    budget = IOLaneBudget(spec_lanes)
    fc = sched = None
    if use_forecast:
        fc = RouterForecast(g)                  # backend mode: GPU routers, no host copies
        sched = SpeculativeScheduler(reader, pool, budget)
    sweep = PrefillSweep(pool, reader, min_tokens=sweep_min_tokens,
                         make_memory=b.make_expert_memory) if sweep_min_tokens else None
    orch = MoEOrchestrator(b, pool, reader, fc, sched, sweep, budget, lookahead=lookahead,
                           lazy_moe=lazy_moe, overlap_resident=overlap_resident)

    def close():
        if sched:
            sched.close()
        if sweep:
            sweep.close()
        reader.close()
        pool.close()
    return orch, close
