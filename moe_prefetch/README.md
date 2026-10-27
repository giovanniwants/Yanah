# moe_prefetch: SlotStream-style expert streaming for Qwen3-Next-80B on an 8 GB Mac

This is the Python memory and IO layer. The math (attention, Gated DeltaNet and
expert matmuls) sits behind the `ModelBackend` protocol in `orchestrator.py`.

Revision 2 rebuilds the scaffold around three mechanisms from
[SlotStream](https://github.com/carloslfu/slotstream) (MIT; see
`THIRD_PARTY_SLOTSTREAM_LICENSE`), scaled down for 8 GB.

```
python -m moe_prefetch.plan             # 8 GB memory plan + tok/s ceiling tables
python -m moe_prefetch.test_synthetic   # end-to-end checks on a toy MoE
```

## Read this first: 15 tok/s on 8 GB

SlotStream's own README says even its smallest memory plan does not fit an
8 GB Mac. Its 16–24 GB tier estimates about 1–6 tok/s for its (larger) model.
`plan.py` applies the same arithmetic to Qwen3-Next.

- **Pool size.** About 4.75 GB of process memory leaves a ~1.8–2.2 GB expert
  pool. That is **21–25 experts per layer out of 512.**
- **Bytes per token.** Every token uses 480 experts at 1.69 MiB each, so an
  all-miss token reads **~810 MiB**.
- **Hit rate needed for 15 tok/s**, with zero compute time:
  - 0.88 at 1.5 GB/s (SlotStream measured this on a base-storage M2 Mac mini)
  - 0.76 at 3 GB/s
  - 0.53 at 6 GB/s

  With a realistic 25–40 ms of compute per token, the needed hit rate rises to
  0.85–0.95.
- **Hit rate you can expect.** SlotStream measured a 0.43 hit rate at ~19
  experts per layer.
- **Realistic ceiling:** roughly 3–6 tok/s on base storage. The full tables are
  in `python -m moe_prefetch.plan`.

None of the mechanisms below changes this. Prefetching and speculation only
hide or share bytes; they don't reduce the bytes an 8 GB cache cannot hold.
The levers that do reduce bytes are fewer bits per expert (3-bit or 2-bit
experts), fewer active experts, or more RAM. If your current ~5 tok/s was
measured on a larger Mac, the OS page cache was probably serving many reads.
Re-measure with `F_NOCACHE` and a memory cap.

## What was adopted from SlotStream

| Mechanism | SlotStream source | Here |
|---|---|---|
| **QD32 demand reads.** Decode misses are latency-bound, so each layer submits its misses as one batch at queue depth 32. Prefill runs at QD12, because those reads are throughput-bound. | `ExpertStore.poolQueueDepth` / `defaultQueueDepth` | `io_lanes.ExpertReader` |
| **Shared lane budget.** Speculative reads pause while a demand batch reads. A forecast that gets demanded is promoted and jumps the queue. | `IOLaneBudget` | `io_lanes.IOLaneBudget` (direct port) |
| **CLOCK slot pool.** One global pool with pins and reference bits. | `SlotPool` | `pool.SlotPool` |
| **Slot adoption.** A forecast reads straight into a reserved slot and joins the cache only when a router demands it. Unused forecasts are discarded, so they never pollute the cache. | `reserveSpeculativeSlot`, `publishSpeculativeSlot` | `pool.reserve_speculative/adopt`, `io_lanes.SpeculativeScheduler` |
| **Router forecast.** Applies the next layer's router to the state right after the current layer's attention. Ships with SlotStream's qualified settings: top 24, probability ≥ 0.062, 32 issued per layer, 1 layer ahead. | `qualifiedDecode`, `EXPERT-LOOKAHEAD.md` | `predictor.RouterForecast` |
| **Scan-resistant prefill sweep.** Passes of 256+ tokens bypass the pool. Experts are processed in groups of 32, sorted by id: resident ones are copied out without touching reference bits, missing ones are read as contiguous runs. Reads for the next group overlap compute on the current one. Only the final prompt pass admits experts, at a per-layer fair share. | `MoELayer.sweep`, `SlotPool.admit` | `sweep.PrefillSweep` |
| **State-recording rollback.** Verify passes record the recurrent state (ssm and conv) after every position. A rejection swaps in the recorded state and trims KV, with no re-run. | `LinearCache.record`, `State.rollback` | `ModelBackend.set_recording/rollback` |
| **Adaptive speculation.** Measures 3 plain steps, then compares the median cost per emitted token. If drafting is >5% slower, it halves the depth; at depth 1 it switches drafting off. | `AdaptiveSpeculation.swift` | `speculative.AdaptiveSpeculationPolicy` |

### Scaled for 8 GB

| Setting | SlotStream | Here |
|---|---|---|
| Speculative lanes | 16 | 4–8 |
| Reserved-slot cap | 64 | 12–24 |
| Headroom floor | 256 slots | 2 layers of routing |
| Router weights | fp32 | fp16 |
| MLX allocator cache cap | 2 GB | 0.5 GB |
| Staging | — | 2 × 32 records (~108 MiB) |

Recording the verify states costs (depth+1) copies of the DeltaNet state. `plan.py` charges for it.

### MTP at K=3 on 8 GB

SlotStream measured MTP as a net slowdown on small caches: ×0.55 at 20 experts
per layer and ×0.96 at 57. It enables MTP only from 76 experts per layer, and
its default depth is 2. The plan here lands at 21–25 per layer, so
`mtp_allowed()` returns False. If you run K=3 anyway, the adaptive policy will
measure it and back off.

## Changes from revision 1

- The asyncio scheduler is gone. The demand path is one blocking batch per
  layer, and speculation uses plain threads with a condition variable. In
  revision 1's test, asyncio added roughly 0.1–0.2 ms of overhead per request,
  and SlotStream's design needs none of it.
- Prefetched data no longer evicts cached experts unless it is actually used.
- The transition-table predictor is dropped. SlotStream found that remembering
  routes failed its adoption gates.
- The test now models a shared bandwidth ceiling for the SSD, so queue depth
  is no longer free.

## Test results (toy model; checks behavior, not real speed)

- **Pool-path forward:** matches a reference that reads weights directly.
- **Prefill sweep:**
  - matches the pool path, apart from summation order;
  - is bit-identical on a cold or a warm pool;
  - non-final passes leave pool contents and reference bits untouched;
  - admission stays within the fair share.
- **Decode with forecasts:** demand misses drop from 88% to 63% of routed
  experts and the pool hit rate rises from 12% to 37%, with output unchanged.
  Forecast recall is 89%. 16 speculative reads were held back behind demand.
- **MTP:** depth 3 and the adaptive policy both reproduce greedy output
  exactly. After rollback, the recurrent state equals plain decoding's at the
  same number of positions. The toy's 100% draft acceptance is an artifact of
  its simple grammar.

Wall-clock gains in the toy are small, because Python overhead dominates
16 KiB reads. Real speed has to be measured on the Mac.

## MLX backend (`mlx_backend.py`)

`build_runtime(model_dir, expert_dir)` wires an mlx-lm Qwen3-Next 4-bit
checkpoint into the orchestrator. Run `repack.py` once first; it writes
`experts.bin` (one aligned record per expert) and `experts.json`.

- **Dense weights are resident.** The model loads lazily, the stacked
  `switch_mlp` experts are removed before evaluation, and only the rest is
  materialized.
- **Experts stream into the pool zero-copy.** The pool is one uint8 MLX array
  per weight piece. `preadv` writes into it through the buffer protocol, and
  `gather_qmm` indexes the slot rows directly with standard affine 4-bit
  dequantization.
- **`moe()` evaluates before returning**, because a demand read may overwrite
  a slot as soon as it is unpinned.
- **Gated DeltaNet rollback.** Verify passes step the recurrence one token at
  a time and record the conv window and recurrent state after each position.
  `rollback(keep)` swaps in position `keep-1` and trims the attention KV
  caches, with no recompute.
- **MTP needs separate weights.** mlx-lm conversions drop the MTP head.
  Without `mtp.*` weights, run with speculation off. The hidden-state
  convention and norm offset are flags to verify against a reference
  implementation.

`test_mlx_backend.py` checks the backend against stock mlx-lm on a tiny
random model. On this CPU build, logits, greedy decode, the rollback state
and GDN stepping all match exactly. On the Mac, run
`backend.check_stepping(layer)` once, because the Metal fused kernel may
differ from per-token stepping by bf16 rounding.

## Revision 3: prefetch changes after the first real run (4.41 tok/s)

**The ceiling these changes can reach.** 4.41 tok/s is 227 ms per token.
With 100 ms of that spent waiting on IO, even perfect prefetching leaves
127 ms, which is **~7.9 tok/s at most**. To pass 10 tok/s, the non-IO time
also has to fall below ~100 ms per token.

**The likely cause of most of the non-IO time.** The first run computed the
router forecast in numpy, against float16 copies of the routers. Numpy has
no fast float16 matmul. At Qwen3-Next's sizes it measured **6.7 ms per layer
here, vs 0.28 ms in float32** (~320 ms per token on this machine; less on
an M1's faster CPU). At lookahead 2 this cost would double, so the forecast
now runs on the GPU instead.

### The four requested changes

| Request | What was built | Expected effect on decode |
|---|---|---|
| Forecast earlier / lookahead 2 | At each layer, the routers for the next two layers run on the GPU in the same sync as the current layer's routing, using the routers already in memory (no host copies, ~100 MB saved). The forecast reads are issued while the current layer's demand reads are still in flight, and the lane budget starts them the moment demand finishes. Extra lead comes from looking two layers ahead, not from reading the state before attention: SlotStream measured that earlier state as a worse predictor (0.62 vs 0.73 agreement). | Two layers of lead for layer l+2. In the toy test, misses fell from 64% to 60% but discarded forecasts tripled. |
| Speculation must not evict residents | A fixed **speculative partition** that no victim scan can see. Unused forecasts return to the partition and evict nothing (tested: 40 wrong forecasts, 0 evictions). An adopted forecast refills the partition with one CLOCK victim, which is exactly the eviction the miss would have caused. Taken literally, with no refill, prefetching starved: forecasts issued fell from 12 to **0** within one decode (tested). | Wrong forecasts no longer push useful experts out of the cache. |
| `DEMAND_QD` 64 | Done. | **None.** A decode layer has at most 10 demand reads (one per record), so 32 and 64 lanes behave the same. SlotStream measured 64 as flat. |
| Sweep threshold 16 | Done. | **None on steady-state decode.** It changes the prompt only: a 25-token chat prompt no longer churns the whole pool, and decode starts on the pool's previous contents plus the prompt's most-used experts. |

### Measuring it on the Mac

`run_mlx_test.py` now reports **decode only** (the old "waited on demand"
included the prompt's reads) and splits each token into GPU attention and
routing, forecast CPU, IO wait, GPU experts, and other. Run these three:

```bash
M=~/Desktop/yanah_models/Qwen3-80B-4bit; X=~/Desktop/yanah_models/Qwen3-80B-4bit-experts
python run_mlx_test.py --model $M --experts $X --tokens 64 --lookahead 0   # forecast cost removed
python run_mlx_test.py --model $M --experts $X --tokens 64 --lookahead 1
python run_mlx_test.py --model $M --experts $X --tokens 64 --lookahead 2
# optional: --forecast-device cpu  (numpy float32 routers, +~196 MiB) to A/B the GPU forecast
```

Use `--tokens 64` or more. In 20 tokens, most of decode happens on a cold
pool. On MLX's CPU backend, each extra router costs ~1.5 ms per layer. On
Metal it should be far cheaper, but that is unmeasured: the `--lookahead 0`
vs `2` comparison measures it directly.

## Revision 4: GPU path (after 6.09 tok/s with lookahead 2)

Measured breakdown per token: 56.1 ms routing sync + 43.0 experts + 52.5 IO
+ 6.0 forecast + 6.5 other = 164 ms.

**Why the GPU time is mostly waiting, not math.** An M1 Max moves ~20 MB of
dense weights per layer in ~0.05 ms. The measured ~2 ms per layer is mostly
round-trip latency: two blocking GPU syncs per layer (96 per token), with
the CPU and GPU taking turns. Stock mlx-lm does one sync per token.

1. **Lazy experts (`mx.eval` removed from the decode path).** That eval was
   a safety fence, not something the math needs. Once a layer's slots are
   unpinned, a pending graph must not read a slot that gets rewritten. But
   the next layer's routing sync already evaluates the whole residual chain,
   and every pool write is made after it. Each layer is now submitted with
   `async_eval` and completes inside the next routing sync: **one sync per
   layer instead of two**. The invariant is enforced explicitly:
   `backend.fence()` runs before every pool-write decision (free in practice).
   The prefill sweep keeps its sync, because its staging buffers are refilled
   while a lazy group would still be pending.
2. **Resident-first overlap.** The dependency chain is strict: layer l's
   experts need their weights, and layer l+1's attention needs layer l's
   output. So the IO wait for layer l's misses can only overlap work that
   doesn't depend on them. In decode, the experts already in memory (hits
   and finished forecasts) and the shared expert are now started on the GPU
   *before* blocking on the misses. The misses run after, and the k columns
   are reassembled in their original order, so the result is bit-identical
   to stock (tested). Loading the *next* layer during this layer's attention
   is what lookahead 2 already does.
3. **`spec_slot_cap` 64** (default in `build_runtime` and `run_mlx_test.py`).
   This only matters if the output shows `refused` forecasts above 0. The
   cost is 57 MB of cache.

Breakdown labels changed: with lazy experts, their GPU time shows up inside
the *next* layer's "routing sync", so compare totals between runs. The A/B
flags are `--sync-moe` (the old two-sync path) and `--no-overlap`.

**Run the parity test on the Mac first.** On CPU, async work and IO
threads rarely overlap in time. On Metal they truly run concurrently, so a
broken invariant would show up there:
`python -m moe_prefetch.test_mlx_backend` checks all three modes against
stock mlx-lm, with a 24-slot pool that evicts on every layer.

## Revision 5: batched decode (`--batch-size N`)

The design mirrors mlx-lm's `BatchGenerator`, which backs its batched server:
- Each prompt is prefilled into its own cache, one prompt at a time, using
  the streamed path (sweep or pool).
- The caches are merged with each cache type's own `merge`: a left-padded
  `BatchKVCache` for the 12 attention layers, and the stacked conv and
  recurrent state for the 36 Gated DeltaNet layers.
- Each decode step is one pass over B rows. Finished sequences are dropped
  with `filter`.

In the expert path, B rows are just a pass of B rows: each layer reads the
**union** of their experts once, and every row uses it. Code:
`MoEOrchestrator.generate_batch`, plus the backend's
`take_cache` / `begin_batch` / `filter_batch` / `end_batch`. Batched tokens
match each sequence decoded alone exactly (test 6: different prompt lengths,
rows filtered mid-run, a pool evicting on every layer).

**Expected expert sharing.** Qwen3-Next routes 10 of 512 experts per
layer, so B unrelated sequences touch `512 * (1 - (1 - 10/512)^B)` unique
experts: 19.8 at B=2 and 38.8 at B=4, which is only **~3% shared** at B=4.
Routing correlation between real prompts can raise that, and the script
measures it: "expert sharing: X unique per layer for B x 10 uses".

**Where batching gains instead.** Batching spreads the per-step fixed
costs over B tokens: GPU round trips, dense attention and DeltaNet weights,
and Python. Expert IO grows almost B-fold. If one sequence costs `F + I` ms
per token (F fixed, I expert IO), a batch step costs about
`F + 3.9 * I` at B=4. At 9.40 tok/s (106 ms), **20 tok/s total at B=4
requires I <= ~32 ms**, and it assumes the pool hit rate survives four
working sets sharing one cache (in the tiny model it fell from 79% to 56%).
Each sequence then runs at ~5 tok/s.

Run it, and compare the same first prompt at batch size 1:
```bash
python run_mlx_test.py --model $M --experts $X --tokens 64 --batch-size 1
python run_mlx_test.py --model $M --experts $X --tokens 64 --batch-size 4
```
Speculative MTP verify is not supported in batch mode.

## Revision 6: testing 4-bit KV and the "semantic lattice"

### 1. KV cache compression (`--kv-bits 4`, `--kv-compare`)
Qwen3-Next keeps a KV cache in only **12 of its 48 layers**: the other 36
are Gated DeltaNet, with a fixed 74 MiB recurrent state. Measured in MLX at
the model's KV shape:

| Context | bf16 KV | 4-bit KV | Freed | = expert slots |
|---:|---:|---:|---:|---:|
| 2,048 | 48 MiB | 13.5 MiB | 34.5 MiB | 20 |
| 4,096 | 96 MiB | 27 MiB | 69 MiB | 41 |
| 8,192 | 192 MiB | 54 MiB | 138 MiB | 82 |
| 32,768 | 768 MiB | 216 MiB | 552 MiB | 327 |

That is +1-3% on a ~1,600-slot pool at chat lengths, and +20% at 32K.
`--kv-compare` measures the quality cost directly: it decodes with bf16
KV, then replays the same tokens with 4-bit KV, and reports top-1
agreement and KL divergence. Use `--prompt-file` with a long document, since
KV quantization error grows with context. 4-bit KV can't be combined with
batching (mlx-lm can't merge quantized caches).

### 2. Semantic lattice (`cache_sim.py`)
Expert ids are per layer, so groups are per layer. Experts are clustered
by co-activation (spectral clustering, default 20 groups). When a token
enters a group (2 or more of its experts in it), the group that most often
follows it at each of the next 3 layers is prefetched whole. Groups are
fitted on 60% of the prompts and scored on the held-out 40%.

`run_mlx_test.py --trace` records real decode routing.
`python -m moe_prefetch.cache_sim TRACE` replays it through the engine's
own CLOCK pool and partition rules. It prints:
- hit rate, demand misses and SSD MiB per token vs pool size, which prices
  the RAM that 4-bit KV frees;
- every prefetch policy side by side;
- a calibration line against the live hit rate. On the tiny model the
  simulator gave 80.0% vs 80.6% live.

**Validation on planted data** (`test_cache_sim.py`, real dimensions):
the lattice cuts demand misses 64% where groups exist vs 13% where they
don't, so the tool can tell the two apart. But even with planted groups,
the lattice alone leaves 4x more demand misses than the router forecast,
and reads 2.4x the SSD bytes. Added on top of the forecast, it removes
~11 more misses per token for ~820 extra reads. A group prefetch reads
~25 experts to catch the ~8 that get used. The router forecast predicts
from the token's own hidden state, so it is more selective.

```bash
python run_mlx_test.py --model $M --experts $X --kv-compare --tokens 64
python run_mlx_test.py --model $M --experts $X --trace ~/Desktop/yanah_build/trace_4bit.npz \
       --trace-prompts 24 --tokens 128          # ~3,000 decode steps; ~6 min at 9 tok/s
python -m moe_prefetch.cache_sim ~/Desktop/yanah_build/trace_4bit.npz
```

## Revision 7: co-activation repacking, measured before rewriting 43 GB

`repack.py` already made the repack SlotStream recommends: each expert's
nine pieces are one contiguous, aligned 1.77 MB record, so each read is
one pread. (SlotStream skipped its own repack because expert-sized reads
reach 55% of SSD peak at queue depth 1 and saturate by about 8.) Co-activation ordering
can only go further by merging *whole records* that sit next to each
other into one larger read. **The bytes stay the same.** Two measurements
decide whether that is worth doing:

1. `python -m moe_prefetch.disk_bench EXPERTS_DIR` (run `sudo purge`
   first). Every read hits fresh offsets, with F_NOCACHE. It prints
   throughput by read size and queue depth, then the **best case**: a
   layer's n misses read scattered (what the engine does) vs perfectly
   adjacent (one pread). If adjacent isn't clearly faster on your SSD,
   no ordering can help.
2. `cache_sim` section [3] (on a `--trace`). It orders each layer's experts
   by co-activation (spectral clusters laid out contiguously, fitted on
   training prompts), then counts how many reads a held-out layer's demand
   misses would really need.

**Validation on planted data:** the ordering came within 1% of an oracle
that knows the true groups. But even the oracle merged only **11%** of
reads without forecasts, and **3%** with the router forecast on. With
forecasting, only ~1.5 demand misses per layer remain, and they land at
random positions inside their group's block.

## Revision 8: model folders, OpenAI server, 2-bit experts

**Folders** (`moe_prefetch/paths.py`; set `YANAH_MODELS` to relocate):
```
~/Desktop/yanah_models/Qwen3-80B-4bit/           dense weights, tokenizer, config (MLX 4-bit)
~/Desktop/yanah_models/Qwen3-80B-4bit-experts/   experts.bin/json (repack.py)
~/Desktop/yanah_models/Qwen3-80B-2bit-experts/   experts.bin/json (quantize_2bit.py)
```
Every script defaults to these.

**`mlx_server.py`** serves the engine at `http://127.0.0.1:8888/v1`:
`POST /chat/completions` (streaming SSE or not), `GET /models`, `GET /health`.
- One engine thread does all model work, and requests queue behind it.
- During a long prefill it sends keepalive comments, so client read
  timeouts don't fire.
- A client disconnect cancels the generation.
- **Conversation reuse.** A request continues the current state if its prompt
  starts with exactly the consumed tokens. Otherwise it rewinds to a snapshot
  taken at the end of the previous prompt (~74 MiB of DeltaNet state is kept
  for that), and reads only the new turn.

`test_mlx_server.py` drives it over real HTTP, with `yanah_shell.py`'s own
stream parser and the official OpenAI client. Turn-2 and turn-3 replies with
reuse were identical to cold rebuilds.

**`quantize_2bit.py`** writes 2-bit g32 experts (AliceAI's scheme; dense
weights stay 4-bit g64 in the original checkpoint). It is resumable, checks
free disk space, and logs the requantization error per layer. The engine on
the result matches stock mlx-lm with the same experts exactly. Measured on
Gaussian weights:

| Experts | Bits/weight incl. scales | Relative weight error | Record |
|---|---:|---:|---:|
| 4-bit g64 (current) | 4.5 | 0.092 | 1.69 MiB |
| 3-bit g64 (`--bits 3 --group-size 64`) | 3.5 | 0.194 | 1.31 MiB |
| 2-bit g32 (default) | 3.0 | 0.361 (0.378 via 4-bit) | 1.125 MiB |

Quantizing via the 4-bit file instead of BF16 adds little error (0.378 vs
0.361).

## Next steps

1. Measure the real disk: never-repeat cold reads with `F_NOCACHE`, as in
   SlotStream's `Tools/coldread.c`.
2. Measure the real fixed memory, and replace every estimate in `plan.py` with
   those numbers.
3. Write the repacker that turns the weights into one record per expert.
4. Implement `ModelBackend` on mlx-lm's Qwen3-Next, including per-position
   recording of the ssm and conv state.
5. Decide on the bytes lever (lower-bit experts or fewer active experts), since
   that is the only path toward 15 tok/s on 8 GB.
