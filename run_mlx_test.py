"""
Warm-up test for the streamed MLX backend on the real Qwen3-Next checkpoint.

  1. Loads dense weights only and reports their real resident memory.
  2. GDN checks on layer 0: fused Metal kernel vs per-token stepping (what
     rollback relies on), and Metal vs CPU math.
  3. Builds the expert pool (explicit --slots, or sized from --budget-gb
     using the MEASURED dense footprint) and reports memory again.
  4. Generates N tokens greedily from a short prompt; prints prefill time,
     decode tok/s (cold first half vs warmer second half) and IO stats.

Run from the folder that contains moe_prefetch/:
    python run_mlx_test.py --budget-gb 5.5        # defaults: moe_prefetch/paths.py
    python run_mlx_test.py --experts ~/Desktop/yanah_models/Qwen3-80B-2bit-experts   # 2-bit experts
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache
from mlx_lm.utils import load_tokenizer

from moe_prefetch.io_lanes import ExpertReader, IOLaneBudget, SpeculativeScheduler
from moe_prefetch.mlx_backend import MLXQwen3NextBackend
from moe_prefetch.orchestrator import MoEOrchestrator
from moe_prefetch.paths import EXPERTS_2BIT, EXPERTS_4BIT, MODEL_4BIT
from moe_prefetch.pool import SlotPool
from moe_prefetch.predictor import RouterForecast
from moe_prefetch.sweep import PrefillSweep

GB = 1e9


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / GB if sys.platform == "darwin" else r * 1024 / GB   # bytes on macOS, KiB on Linux


def mem(label: str) -> float:
    mx.clear_cache()
    a = mx.get_active_memory() / GB
    print(f"  [{label}] MLX active {a:.2f} GB | MLX peak {mx.get_peak_memory() / GB:.2f} GB "
          f"| process peak RSS {peak_rss_gb():.2f} GB")
    return a


def gdn_checks(b: MLXQwen3NextBackend, layer: int, T: int = 4) -> None:
    d_step = b.check_stepping(layer, T=T)
    print(f"  fused kernel vs per-token stepping, layer {layer}: max |diff| = {d_step:.3e}"
          f"  {'OK' if d_step < 1e-3 else 'WARNING: rollback state will drift from the fused path'}")

    if not mx.metal.is_available():
        print("  Metal vs CPU: skipped (no Metal on this machine)")
        return
    # Metal vs CPU: same layer, same input, fresh caches.
    m = b.model.layers[layer].linear_attn
    x = mx.random.normal((1, T, b.args.hidden_size), key=mx.random.key(1)).astype(
        b.model.model.norm.weight.dtype)
    outs = []
    for dev in (mx.gpu, mx.cpu):
        prev = mx.default_device()
        try:
            mx.set_default_device(dev)
            c = ArraysCache(2)
            y = b._gated_delta(m, x, c, layer)
            mx.eval(y, c[1])
            outs.append((y.astype(mx.float32), c[1]))
        finally:
            mx.set_default_device(prev)
    dy = mx.abs(outs[0][0] - outs[1][0]).max().item()
    ds = mx.abs(outs[0][1] - outs[1][1]).max().item()
    scale = mx.abs(outs[1][0]).max().item()
    print(f"  Metal vs CPU, layer {layer}: output max |diff| = {dy:.3e} (output scale {scale:.2e}), "
          f"state max |diff| = {ds:.3e}  (bf16 rounding is expected, not exactness)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_4BIT),
                    help=f"MLX Qwen3-Next checkpoint dir (default {MODEL_4BIT})")
    ap.add_argument("--experts", default=str(EXPERTS_4BIT),
                    help=f"experts.bin/json dir (default {EXPERTS_4BIT}; 2-bit: {EXPERTS_2BIT})")
    ap.add_argument("--slots", type=int, default=None, help="pool slots (overrides --budget-gb)")
    ap.add_argument("--budget-gb", type=float, default=5.5, help="process memory target")
    ap.add_argument("--reserve-gb", type=float, default=0.9,
                    help="non-pool overhead besides dense weights: KV, GDN state, MLX cache cap, "
                         "activations, sweep staging, forecast routers")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    ap.add_argument("--prompt-file", default=None,
                    help="read the prompt from a file (e.g. a long document, to test KV at long context)")
    ap.add_argument("--kv-bits", type=int, default=None, choices=(4, 8),
                    help="quantize the 12 attention layers' KV cache (mlx-lm QuantizedKVCache)")
    ap.add_argument("--kv-group-size", type=int, default=64, help=argparse.SUPPRESS)
    ap.add_argument("--kv-compare", action="store_true",
                    help="quality check: decode with bf16 KV, then replay the same tokens with "
                         "--kv-bits (default 4) KV; report top-1 agreement and KL")
    ap.add_argument("--trace", default=None,
                    help="record decode routing to this .npz for `python -m moe_prefetch.cache_sim`")
    ap.add_argument("--trace-prompts", type=int, default=24,
                    help="prompts to decode for --trace (each --tokens long)")
    ap.add_argument("--prompt-ids", default=None,
                    help="comma-separated token ids; skips the tokenizer (smoke tests)")
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--lookahead", type=int, default=2, help="forecast layers l+1 .. l+N (0 = issue none)")
    ap.add_argument("--forecast-device", choices=("gpu", "cpu"), default="gpu",
                    help="gpu: model's own 8-bit routers in the routing sync (no extra memory); "
                         "cpu: float32 numpy copies of the routers (+~196 MiB host memory)")
    ap.add_argument("--spec-slots", type=int, default=None,
                    help="speculative partition size; default 64, or 32 x batch size "
                         "(only matters if 'refused' forecasts > 0)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="decode this many different prompts as one batch (expert reads shared)")
    ap.add_argument("--prompts-file", default=None,
                    help="one prompt per line for --batch-size > 1 (default: built-in varied prompts)")
    ap.add_argument("--sync-moe", action="store_true",
                    help="old behaviour: block on every layer's experts (mx.eval per layer)")
    ap.add_argument("--no-overlap", action="store_true",
                    help="don't start in-memory experts on the GPU before the IO wait")
    ap.add_argument("--sweep-min", type=int, default=16, help="prompt passes >= this use the sweep")
    ap.add_argument("--cache-limit-gb", type=float, default=0.5)
    args = ap.parse_args()
    spec_auto = args.spec_slots is None
    model_dir, exp_dir = str(Path(args.model).expanduser()), str(Path(args.experts).expanduser())

    print(f"device: {mx.default_device()} | Metal available: {mx.metal.is_available()}")
    mx.reset_peak_memory()
    base = mem("start")

    # 1. dense weights ----------------------------------------------------
    t0 = time.perf_counter()
    b = MLXQwen3NextBackend(model_dir, exp_dir, cache_limit_gb=args.cache_limit_gb, load_mtp=False,
                            kv_bits=None if args.kv_compare else args.kv_bits,
                            kv_group_size=args.kv_group_size)
    print(f"\n[1] dense weights loaded in {time.perf_counter() - t0:.1f} s")
    dense_params = sum(v.nbytes for _, v in tree_flatten(b.model.parameters())) / GB
    dense_active = mem("after dense load") - base
    g = b.geometry
    print(f"  dense parameter bytes {dense_params:.2f} GB (plan.py estimate: 1.50 GB)")
    print(f"  geometry: {g.n_layers} layers x {g.n_experts} experts, top-{g.top_k}, "
          f"{g.bits}-bit g{g.group_size}, record {g.expert_nbytes / 2**20:.2f} MiB, MTP layers {g.n_mtp_layers}")

    # 2. GDN checks ---------------------------------------------------------
    print("\n[2] Gated DeltaNet checks")
    gdn_checks(b, layer=0)

    # 3. pool ---------------------------------------------------------------
    if args.slots is None:
        pool_gb = args.budget_gb - dense_active - args.reserve_gb
        slots = int(pool_gb * GB // g.slot_nbytes)
        print(f"\n[3] pool from budget: {args.budget_gb:.2f} - dense {dense_active:.2f} "
              f"- reserve {args.reserve_gb:.2f} = {pool_gb:.2f} GB")
    else:
        slots = args.slots
        print("\n[3] pool from --slots")
    print(f"  {slots} slots = {slots / g.total_moe_layers:.1f} experts/layer "
          f"({slots * g.slot_nbytes / GB:.2f} GB)")
    if slots < 6 * g.top_k:
        sys.exit("pool too small for this budget; raise --budget-gb or pass --slots")

    if spec_auto:   # 64, or 32 per row for a batch; never more than a quarter of the pool
        args.spec_slots = min(64 if args.batch_size <= 1 else max(64, 32 * args.batch_size),
                              max(1, slots // 4))
    pool = SlotPool(g, slots, spec_slot_cap=args.spec_slots, memory=b.make_expert_memory(slots))
    reader = ExpertReader(b.layout())
    budget = IOLaneBudget(8)
    fc = sched = None
    if not args.no_forecast:
        if args.forecast_device == "gpu":
            fc = RouterForecast(g)     # backend mode: the model's own routers, on the GPU
        else:
            gates, norms = b.forecast_weights()
            fc = RouterForecast(g, gates, norms)   # numpy float32 (never float16: ~24x slower)
            print(f"  numpy router copies: {fc.resident_bytes / 2**20:.0f} MiB")
        sched = SpeculativeScheduler(reader, pool, budget)
        if args.batch_size > 1:
            # B rows route ~B x 10 experts per layer; SlotStream's caps (24
            # candidates, 32 issued) were tuned for one row.
            sched.top = max(24, 12 * args.batch_size)
            sched.issue_cap = max(32, 12 * args.batch_size)
    sweep = PrefillSweep(pool, reader, min_tokens=args.sweep_min, make_memory=b.make_expert_memory)
    orch = MoEOrchestrator(b, pool, reader, fc, sched, sweep, budget, lookahead=args.lookahead,
                           lazy_moe=not args.sync_moe, overlap_resident=not args.no_overlap)
    print(f"  forecast {'off' if args.no_forecast else f'lookahead {args.lookahead}, on {args.forecast_device.upper()}'}, "
          f"spec partition {args.spec_slots} slots, sweep >= {args.sweep_min} tokens, "
          f"demand QD {reader._demand._max_workers}")
    print(f"  experts: {'lazy, 1 GPU sync per layer' if orch.lazy else 'synced, 2 GPU syncs per layer'}"
          f"{', resident-first overlap' if orch.overlap else ''}")
    mem("after pool + staging")

    # 4. generate -----------------------------------------------------------
    if args.kv_compare or args.trace:
        tok = load_tokenizer(Path(model_dir)) if not args.prompt_ids else None
        if args.kv_compare:
            run_kv_compare(args, b, orch, tok)
        else:
            run_trace(args, b, g, orch, reader, pool, fc, tok)
        if sched:
            sched.close()
        sweep.close(); reader.close(); pool.close()
        return
    if args.batch_size > 1:
        run_batch(args, b, g, orch, reader, pool, fc, model_dir)
        if sched:
            sched.close()
        sweep.close(); reader.close(); pool.close()
        return
    if args.prompt_ids:
        tok, ids, eos = None, [int(t) for t in args.prompt_ids.split(",")], set()
    else:
        tok = load_tokenizer(Path(model_dir))
        ids, eos = chat_ids(tok, prompt_text(args)), eos_of(tok)
    print(f"\n[4] prompt {len(ids)} tokens, generating {args.tokens}"
          f"{f', KV {args.kv_bits}-bit' if args.kv_bits else ''}")

    t0 = time.perf_counter()
    logits, _ = orch.prefill(ids, 256)
    t_prefill = time.perf_counter() - t0
    import copy
    rs0, ps0, os0 = copy.copy(reader.stats), copy.copy(pool.stats), orch.stats.snapshot()
    fc0 = copy.copy(fc.stats) if fc else None
    out, step_s = [], []
    for _ in range(args.tokens):
        nxt = int(np.argmax(logits[-1]))
        out.append(nxt)
        if nxt in eos:
            break
        t = time.perf_counter()
        logits, _ = orch.forward([nxt])
        step_s.append(time.perf_counter() - t)

    print(f"  text: {tok.decode(out)!r}" if tok else f"  ids: {out}")
    print(f"  prefill {t_prefill:.2f} s ({len(ids) / t_prefill:.1f} tok/s)")
    if step_s:
        h = len(step_s) // 2 or 1
        print(f"  decode {len(step_s)} steps: {len(step_s) / sum(step_s):.2f} tok/s overall | "
              f"first half {h / sum(step_s[:h]):.2f} tok/s, second half "
              f"{(len(step_s) - h) / max(1e-9, sum(step_s[h:])):.2f} tok/s | "
              f"median step {np.median(step_s) * 1e3:.0f} ms")
    rs, ps = reader.stats, pool.stats
    print(f"  prompt IO: {rs0.bytes / GB:.2f} GB ({rs0.sweep_records} sweep, {rs0.demand_records} demand records)")
    n = max(1, len(step_s))
    d = lambda a, b: a - b
    print(f"  DECODE ONLY, per token:")
    print(f"    time   {orch.stats.breakdown(since=os0, per=n)}")
    if orch.lazy:
        print("           (lazy experts: their GPU time is inside the NEXT layer's routing sync; compare totals)")
    rec = d(rs.demand_records, rs0.demand_records) + d(rs.spec_records, rs0.spec_records)
    print(f"    reads  {d(rs.bytes, rs0.bytes) / n / 2**20:.0f} MiB ({d(rs.demand_records, rs0.demand_records) / n:.1f} demand "
          f"+ {d(rs.spec_records, rs0.spec_records) / n:.1f} forecast records of {g.n_layers * g.top_k} expert uses)")
    io_s = orch.stats.t_io - os0.t_io
    if io_s > 0 and rec:
        print(f"    disk   {d(rs.bytes, rs0.bytes) / io_s / GB:.2f} GB/s effective over io-wait time "
              f"(demand reads alone {d(rs.demand_records, rs0.demand_records) * g.expert_nbytes / max(1e-9, d(rs.demand_wait_s, rs0.demand_wait_s)) / GB:.2f} GB/s)")
    hits, miss = d(ps.hits, ps0.hits), d(ps.misses, ps0.misses)
    print(f"    pool   hit {hits / max(1, hits + miss):.0%} | adopted {d(ps.adopted, ps0.adopted) / n:.1f}, "
          f"discarded {d(ps.spec_discarded, ps0.spec_discarded) / n:.1f}, refused {d(ps.spec_refused, ps0.spec_refused) / n:.1f} "
          f"forecasts | evictions {d(ps.evictions, ps0.evictions) / n:.1f}")
    if fc:
        cov, act = fc.stats.covered - fc0.covered, fc.stats.actual - fc0.actual
        print(f"    forecast recall {cov / max(1, act):.0%} (union of the l+1..l+{args.lookahead} forecasts)")
    print(f"  whole run: {orch.stats.summary()}")
    print(f"  KV cache now {b.kv_nbytes() / 2**20:.1f} MiB "
          f"({'bf16' if not args.kv_bits else f'{args.kv_bits}-bit'}, allocated in 256-token steps)")
    mem("end")

    if sched:
        sched.close()
    sweep.close()
    reader.close()
    pool.close()


DEFAULT_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that checks whether a string is a palindrome.",
    "List three practical tips for keeping houseplants alive in winter.",
    "Summarize the causes of the French Revolution in one short paragraph.",
    "What is the difference between TCP and UDP? Answer briefly.",
    "Give me a haiku about autumn rain.",
    "How does a refrigerator keep food cold? Explain simply.",
    "Translate 'good morning, how are you?' into Spanish and French.",
    "Write a SQL query that returns the ten customers with the highest total order value.",
    "What are the main differences between mitosis and meiosis?",
    "Draft a polite email asking a landlord to fix a leaking kitchen tap.",
    "Explain what a hash map is to a beginner programmer.",
    "Give a short history of the printing press and why it mattered.",
    "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed?",
    "Write a short bedtime story about a fox who is afraid of the dark.",
    "What should I consider when choosing between renting and buying a home?",
    "Explain how vaccines train the immune system, in plain language.",
    "Write a bash one-liner that counts the lines in every .py file in a folder.",
    "Compare the climates of Norway and Spain in a few sentences.",
    "What is the Pythagorean theorem, and where is it used in real life?",
    "Suggest a three-day itinerary for a first visit to Rome.",
    "Explain the difference between a virus and a bacterium.",
    "Write a JavaScript function that debounces another function.",
    "Why do leaves change colour in autumn?",
]


def prompt_text(args) -> str:
    return open(Path(args.prompt_file).expanduser()).read() if args.prompt_file else args.prompt


def chat_ids(tok, text: str) -> list:
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def eos_of(tok) -> set:
    return set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def run_kv_compare(args, b, orch, tok) -> None:
    """Teacher-forced: the quantized-KV run is fed the SAME tokens the bf16
    run produced, so every position is compared under identical context
    (no divergence cascade). Reports top-1 agreement and KL(bf16 || quantized)."""
    bits = args.kv_bits or 4
    ids = [int(t) for t in args.prompt_ids.split(",")] if args.prompt_ids else chat_ids(tok, prompt_text(args))
    eos = set() if tok is None else eos_of(tok)
    print(f"\n[4] KV quality: prompt {len(ids)} tokens, {args.tokens} decode positions, bf16 vs {bits}-bit KV")
    runs = {}
    for kvb in (None, bits):
        b.kv_bits = kvb
        b.reset()
        if orch.fc is not None:
            orch.fc.reset()
        logits, _ = orch.prefill(ids, 256)
        seq_logits = [logits[-1].copy()]
        if kvb is None:
            toks = []
            for _ in range(args.tokens):
                t = int(np.argmax(seq_logits[-1])); toks.append(t)
                if t in eos or len(toks) == args.tokens:
                    break
                lg, _ = orch.forward([t]); seq_logits.append(lg[-1].copy())
        else:
            for t in toks[:len(runs[None][0]) - 1]:
                lg, _ = orch.forward([t]); seq_logits.append(lg[-1].copy())
        runs[kvb] = (seq_logits, b.kv_nbytes())
    base, quant = runs[None][0], runs[bits][0]
    agree, kls, dmax = 0, [], 0.0
    for a, q in zip(base, quant):
        agree += int(np.argmax(a) == np.argmax(q))
        la, lq = _log_softmax(a), _log_softmax(q)
        kls.append(float((np.exp(la) * (la - lq)).sum()))
        dmax = max(dmax, float(np.abs(a - q).max()))
    n = len(base)
    ctx = len(ids) + n
    print(f"  top-1 agreement {agree}/{n} = {agree / n:.1%} | KL(bf16||{bits}-bit) mean {np.mean(kls):.2e}, "
          f"max {np.max(kls):.2e} nats | max |logit diff| {dmax:.3f}")
    print(f"  KV at {ctx} tokens: bf16 {runs[None][1] / 2**20:.1f} MiB -> {bits}-bit {runs[bits][1] / 2**20:.1f} MiB "
          f"(allocated in 256-token steps)")
    if tok is not None:
        print(f"  bf16 text: {tok.decode(toks)[:120]!r}")
    b.kv_bits = None


def run_trace(args, b, g, orch, reader, pool, fc, tok) -> None:
    """Decode --trace-prompts different prompts one after another (the pool
    stays warm, as in a real session) and record every decode step's
    routing and forecast candidates."""
    import copy
    from moe_prefetch.cache_sim import save_trace
    texts = ([l.strip() for l in open(Path(args.prompts_file).expanduser()) if l.strip()]
             if args.prompts_file else DEFAULT_PROMPTS)[:args.trace_prompts]
    out = str(Path(args.trace).expanduser())
    print(f"\n[4] tracing {len(texts)} prompts x up to {args.tokens} tokens -> {out}")
    orch.trace = []
    pog, steps, t0 = {}, 0, time.perf_counter()
    rs0 = ps0 = os0 = None
    for i, text in enumerate(texts):
        if args.prompt_ids:
            base = [int(t) for t in args.prompt_ids.split(",")]
            ids, eos = base[i % len(base):] + base[:i % len(base)], set()
        else:
            ids, eos = chat_ids(tok, text), eos_of(tok)
        b.reset()
        if fc is not None:
            fc.reset()
        logits, _ = orch.prefill(ids, 256)
        if rs0 is None:     # decode-only stats start after the first prompt pass
            rs0, ps0, os0 = copy.copy(reader.stats), copy.copy(pool.stats), orch.stats.snapshot()
        for _ in range(args.tokens - 1):
            t = int(np.argmax(logits[-1]))
            if t in eos:
                break
            logits, _ = orch.forward([t])
            pog[orch.generation] = i
            steps += 1
        print(f"  prompt {i + 1}/{len(texts)} done, {steps} decode steps, {time.perf_counter() - t0:.0f} s")
    rs, ps, st = reader.stats, pool.stats, orch.stats
    hits, miss = ps.hits - ps0.hits, ps.misses - ps0.misses
    io_s = st.t_io - os0.t_io
    gbps = (rs.bytes - rs0.bytes) / io_s / 1e9 if io_s > 0 else None
    meta = dict(n_layers=g.n_layers, n_experts=g.n_experts, top_k=g.top_k, slots=pool.slots,
                spec_slots=pool.spec_slot_cap, record_bytes=g.expert_nbytes,
                live_hit_rate=hits / max(1, hits + miss), disk_gbps=gbps,
                lookahead=orch.lookahead, tokens_per_prompt=args.tokens)
    save_trace(out, orch.trace, pog, meta)
    orch.trace = None
    print(f"  saved {steps} steps | live pool hit {meta['live_hit_rate']:.1%}"
          f"{f', effective disk {gbps:.2f} GB/s' if gbps else ''}")
    print(f"  next: python -m moe_prefetch.cache_sim {out}")


def expected_unique(n_experts: int, top_k: int, rows: int) -> float:
    """Unique experts per layer if the rows routed independently at random."""
    return n_experts * (1 - (1 - top_k / n_experts) ** rows)


def run_batch(args, b, g, orch, reader, pool, fc, model_dir) -> None:
    import copy
    B = args.batch_size
    if args.prompts_file:
        texts = [l.strip() for l in open(Path(args.prompts_file).expanduser()) if l.strip()]
    else:
        texts = DEFAULT_PROMPTS
    if len(texts) < B:
        sys.exit(f"need {B} prompts, have {len(texts)}")
    texts = texts[:B]
    if args.prompt_ids:
        tok, eos = None, set()
        base = [int(t) for t in args.prompt_ids.split(",")]
        prompts = [base[i % len(base):] + base[:i % len(base)] for i in range(B)]   # B different rotations
    else:
        tok = load_tokenizer(Path(model_dir))
        prompts = []
        for t in texts:
            ids = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True)
            prompts.append(list(ids["input_ids"] if isinstance(ids, dict) else ids))
        eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    print(f"\n[4] batch of {B} different prompts ({', '.join(str(len(p)) for p in prompts)} tokens), "
          f"up to {args.tokens} tokens each")

    snap = {}
    def mark():
        snap.update(rs=copy.copy(reader.stats), ps=copy.copy(pool.stats), os=orch.stats.snapshot(),
                    t=time.perf_counter())
    t0 = time.perf_counter()
    steps: list = []
    outs = orch.generate_batch(prompts, args.tokens, eos, 256, step_log=steps, on_decode_start=mark)
    t_end = time.perf_counter()
    for i, o in enumerate(outs):
        txt = tok.decode(o) if tok else str(o)
        print(f"  [{i}] {txt[:90]!r}{'...' if len(txt) > 90 else ''}")
    if not steps:
        print("  every sequence ended at its first token"); return

    rs0, ps0, os0 = snap["rs"], snap["ps"], snap["os"]
    rs, ps, st = reader.stats, pool.stats, orch.stats
    dec_s = sum(s for _, s in steps)
    toks = sum(r for r, _ in steps)
    full = [s for r, s in steps if r == B]
    print(f"  prompts (one at a time) {snap['t'] - t0:.2f} s")
    print(f"  DECODE: {toks} tokens in {dec_s:.2f} s = {toks / dec_s:.2f} tok/s TOTAL")
    if full:
        print(f"    with all {B} rows active: {len(full)} steps, median step {np.median(full) * 1e3:.0f} ms "
              f"= {B / np.median(full):.2f} tok/s total, {1 / np.median(full):.2f} tok/s per sequence")
    n = len(steps)
    print(f"    per step: {st.breakdown(since=os0, per=n)}")
    layers = st.pool_layers - os0.pool_layers
    uniq = (st.routed - os0.routed) / max(1, layers)
    rows = toks / n
    print(f"    expert sharing: {uniq:.1f} unique experts per layer per step for {rows:.2f} rows x "
          f"{g.top_k} = {rows * g.top_k:.1f} expert uses (independent routing would give "
          f"{expected_unique(g.n_experts, g.top_k, round(rows)):.1f})")
    d = lambda a, c: a - c
    print(f"    reads per step {d(rs.bytes, rs0.bytes) / n / 2**20:.0f} MiB "
          f"({d(rs.demand_records, rs0.demand_records) / n:.1f} demand + "
          f"{d(rs.spec_records, rs0.spec_records) / n:.1f} forecast records); per token "
          f"{d(rs.bytes, rs0.bytes) / toks / 2**20:.0f} MiB")
    hits, miss = d(ps.hits, ps0.hits), d(ps.misses, ps0.misses)
    print(f"    pool hit {hits / max(1, hits + miss):.0%} | forecasts refused {d(ps.spec_refused, ps0.spec_refused) / n:.1f}"
          f"/step | evictions {d(ps.evictions, ps0.evictions) / n:.1f}/step")
    mem("end")


if __name__ == "__main__":
    main()
