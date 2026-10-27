"""
OpenAI-compatible server for the streamed Qwen3-Next engine (moe_prefetch).

    POST /v1/chat/completions   stream (SSE) or not; temperature, top_p,
                                top_k, max_tokens, stop, seed
    GET  /v1/models             the loaded model
    GET  /health                200 once the model is loaded

    cd ~/Desktop/yanah_build && source .venv/bin/activate
    python mlx_server.py                        # 4-bit, port 8888, 5.5 GB budget
    python mlx_server.py --experts ~/Desktop/yanah_models/Qwen3-80B-2bit-experts

Point Yanah at it with YANAH_BASE_URL=http://127.0.0.1:8888/v1.

Design
------
* One engine thread does ALL model work (load, prefill, decode). HTTP
  handlers queue a job and relay its tokens, so requests never interleave
  inside the model; a second request waits its turn.
* Streaming sends SSE `data:` chunks in OpenAI's format, ending with
  `data: [DONE]`. While a long prompt is being read, it sends SSE comment
  lines (": keepalive"), which clients skip, so a 300 s read timeout doesn't
  fire during prefill. If the client disconnects, the job is cancelled at the
  next token.
* Conversation prefix reuse, two levels:
  1. The prompt starts with EXACTLY the tokens the current state consumed
     (prompt + reply): continue from there. The consumed-token list is
     tracked, never inferred: the last sampled token is not fed.
  2. Otherwise, if it starts with the previous PROMPT: restore a snapshot
     taken at the end of that prompt and read only the rest. The chat
     template re-tokenizes the reply from text, which need not reproduce
     the generated ids, so this is the level a chat turn usually hits.
     Gated DeltaNet state can't be rewound, so the snapshot just keeps
     references to the prompt-end state (~74 MiB). KV caches trim back.
  Anything else rebuilds from scratch. A continued state can differ from a
  cold rebuild by float rounding (SlotStream measured this), never by
  content. Disable with --no-prefix-reuse.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context

from moe_prefetch.paths import EXPERTS_4BIT, MODEL_4BIT
from tool_calls import ToolCallStream, normalize_messages

GB = 1e9


# ---------------------------------------------------------------- sampling
def sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int,
           rng: np.random.Generator) -> int:
    """Greedy at temperature 0; otherwise temperature, top-k, then top-p
    (nucleus). Drawn against the unnormalized CDF, so it always lands on a
    token with mass and never divides 0/0."""
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    order = np.argsort(-p)
    if 0 < top_k < len(order):
        order = order[:top_k]
    probs = p[order]
    if 0 < top_p < 1:
        c = np.cumsum(probs)
        keep = int(np.searchsorted(c, top_p * c[-1])) + 1
        order, probs = order[:keep], probs[:keep]
    c = np.cumsum(probs)
    return int(order[min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(order) - 1)])


@dataclass
class Params:
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    stop: List[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class Job:
    messages: list
    params: Params
    tools: Optional[list] = None
    anchor: Optional[str] = None       # "cache_anchor": keep a snapshot where this text ends
    out: "queue.Queue" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)


class StopMatcher:
    """Withholds text that could still turn into a stop string."""

    def __init__(self, stops: List[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""

    def push(self, text: str):
        """-> (text safe to emit, stopped?)"""
        self.buf += text
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0:
                out, self.buf = self.buf[:i], ""
                return out, True
        hold = 0
        for s in self.stops:
            for n in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:n]):
                    hold = max(hold, n)
                    break
        out = self.buf[:len(self.buf) - hold] if hold else self.buf
        self.buf = self.buf[len(out):]
        return out, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, args):
        self.args = args
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.ready = threading.Event()
        self.load_error: Optional[str] = None
        self.model_id = args.model_id or Path(args.model).name
        # --system-prompt-file: text put at the start of every request's system prompt (YanahCoder)
        self.system_prompt = (Path(args.system_prompt_file).expanduser().read_text(encoding="utf-8").strip()
                              if getattr(args, "system_prompt_file", None) else "")
        self.busy = False
        self.last: dict = {}             # stats of the last finished request (shown in /health)
        self.thread = threading.Thread(target=self._run, name="mlx-engine", daemon=True)
        self.thread.start()

    # -- everything below runs on the engine thread --------------------------
    def _load(self) -> None:
        import mlx.core as mx
        from mlx_lm.utils import load_tokenizer
        from moe_prefetch.io_lanes import ExpertReader, IOLaneBudget, SpeculativeScheduler
        from moe_prefetch.mlx_backend import MLXQwen3NextBackend
        from moe_prefetch.orchestrator import MoEOrchestrator
        from moe_prefetch.pool import SlotPool
        from moe_prefetch.predictor import RouterForecast
        from moe_prefetch.sweep import PrefillSweep

        a = self.args
        mx.clear_cache()
        base = mx.get_active_memory()
        t0 = time.perf_counter()
        b = MLXQwen3NextBackend(str(Path(a.model).expanduser()), str(Path(a.experts).expanduser()),
                                cache_limit_gb=a.cache_limit_gb, load_mtp=False)
        mx.clear_cache()
        dense = (mx.get_active_memory() - base) / GB
        g = b.geometry
        slots = a.slots or int((a.budget_gb - dense - a.reserve_gb) * GB // g.slot_nbytes)
        if slots < 6 * g.top_k:
            raise RuntimeError(f"budget {a.budget_gb} GB leaves {slots} expert slots; raise --budget-gb")
        spec = min(a.spec_slots, max(1, slots // 4))
        pool = SlotPool(g, slots, spec_slot_cap=spec, memory=b.make_expert_memory(slots))
        reader = ExpertReader(b.layout())
        budget = IOLaneBudget(8)
        fc, sched = RouterForecast(g), SpeculativeScheduler(reader, pool, budget)
        sweep = PrefillSweep(pool, reader, min_tokens=16, make_memory=b.make_expert_memory)
        self.orch = MoEOrchestrator(b, pool, reader, fc, sched, sweep, budget, lookahead=a.lookahead)
        self.b, self.fc, self.pool = b, fc, pool
        self.tok = load_tokenizer(Path(a.model).expanduser())
        eos = set(self.tok.eos_token_ids)
        try:
            gc = json.load(open(Path(a.model).expanduser() / "generation_config.json"))
            e = gc.get("eos_token_id", [])
            eos |= set(e if isinstance(e, list) else [e])
            self.defaults = Params(temperature=gc.get("temperature", 0.7), top_p=gc.get("top_p", 1.0),
                                   top_k=gc.get("top_k", 0))
        except (OSError, ValueError):
            self.defaults = Params()
        self.eos = eos
        self.held: List[int] = []        # token ids the model state has consumed
        self.snap = None                 # (prompt ids, backend snapshot at the end of that prompt)
        self.anchor = None               # (ids, snapshot) where the request's cache_anchor text ends
        print(f"[mlx_server] {self.model_id}: dense {dense:.2f} GB, {slots} expert slots "
              f"({slots / g.total_moe_layers:.0f}/layer, {slots * g.slot_nbytes / GB:.2f} GB, "
              f"{g.bits}-bit experts), loaded in {time.perf_counter() - t0:.0f} s", flush=True)

    def _run(self) -> None:
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"[mlx_server] load failed: {self.load_error}", flush=True)
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.jobs.get()
            self.busy = True
            try:
                self._generate(job)
            except Exception as e:  # noqa: BLE001
                self.held, self.snap, self.anchor = [], None, None   # state may be half-updated: rebuild next time
                job.out.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                self.busy = False

    def prompt_ids(self, messages, tools=None) -> List[int]:
        kw = dict(add_generation_prompt=True)
        if tools:
            kw["tools"] = tools
        ids = self.tok.apply_chat_template(messages, **kw)
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    def anchor_len(self, messages, tools, ids: List[int], anchor: str) -> int:
        """How many prompt tokens come before the end of `anchor` (text inside the prompt, e.g.
        the part of the system prompt every persona shares), minus a 2-token margin so the
        boundary token can't depend on the text after it. 0 if the text isn't in the prompt."""
        kw = dict(add_generation_prompt=True, tokenize=False)
        if tools:
            kw["tools"] = tools
        text = self.tok.apply_chat_template(messages, **kw)
        k = text.find(anchor) if isinstance(text, str) else -1
        if k < 0:
            return 0
        pre = self.tok.encode(text[:k + len(anchor)], add_special_tokens=False)
        n, m = 0, min(len(pre), len(ids))
        while n < m and pre[n] == ids[n]:
            n += 1
        return max(0, n - 2)

    def _generate(self, job: Job) -> None:
        a, o, p = self.args, self.orch, job.params
        ids = self.prompt_ids(job.messages, job.tools)
        if len(ids) + 1 > a.max_context:
            raise ValueError(f"prompt is {len(ids)} tokens; --max-context is {a.max_context}")
        max_new = max(1, min(p.max_tokens, a.max_context - len(ids)))
        t0 = time.perf_counter()
        n = 0
        if not a.no_prefix_reuse and self.held and len(ids) > len(self.held) \
                and ids[:len(self.held)] == self.held:
            n = len(self.held)                                   # level 1: continue
        elif not a.no_prefix_reuse and self.snap and len(ids) > len(self.snap[0]) \
                and ids[:len(self.snap[0])] == self.snap[0]:
            self.b.restore(self.snap[1])                         # level 2: rewind to prompt end
            n = len(self.snap[0])
        how = "continue" if n and self.held and n == len(self.held) else ("prompt" if n else "")
        if n == 0 and not a.no_prefix_reuse and self.anchor and len(ids) > len(self.anchor[0]) \
                and ids[:len(self.anchor[0])] == self.anchor[0]:
            self.b.restore(self.anchor[1])                       # level 3: rewind to the anchor
            n, how = len(self.anchor[0]), "anchor"               # (e.g. persona switch: shared part kept)
        if n == 0:
            self.b.reset()
            self.fc.reset()
            self.anchor = None                                   # its cache is gone
        feed = ids[n:]
        self.held, self.snap = [], None  # invalid until this request finishes cleanly
        cut = 0
        if not a.no_prefix_reuse and job.anchor and self.anchor is None:
            try:
                cut = self.anchor_len(job.messages, job.tools, ids, job.anchor)
            except Exception as e:  # noqa: BLE001  (never let the anchor break a reply)
                print(f"[mlx_server] cache_anchor ignored: {type(e).__name__}: {e}", flush=True)
                cut = 0
        if n < cut < len(ids):
            o.prefill(ids[n:cut], a.prefill_chunk)               # up to the anchor, then snapshot
            self.anchor = (list(ids[:cut]), self.b.snapshot())
            logits, _ = o.prefill(ids[cut:], a.prefill_chunk)
        else:
            logits, _ = o.prefill(feed, a.prefill_chunk)
        consumed = list(ids)
        if not a.no_prefix_reuse:
            self.snap = (list(ids), self.b.snapshot())
        t1 = time.perf_counter()
        rng = np.random.default_rng(p.seed)
        detok = self.tok.detokenizer
        stopper = StopMatcher(p.stop)
        gen, finish, stopped = 0, "length", False
        while True:
            if job.cancel.is_set():
                finish = "cancelled"
                break
            t = sample(logits[-1], p.temperature, p.top_p, p.top_k, rng)
            if t in self.eos:
                finish = "stop"
                break
            gen += 1
            detok.add_token(t)
            text, stopped = stopper.push(detok.last_segment)
            if text:
                job.out.put(("delta", text))
            if stopped:                  # a stop string: nothing after it is sent
                finish = "stop"
                break
            if gen >= max_new:
                break                    # sampled but not fed: not part of the state
            logits, _ = o.forward([t])
            consumed.append(t)
        if not stopped:
            detok.finalize()             # bytes held back for a split UTF-8 character
            tail, stopped = stopper.push(detok.last_segment)
            if not stopped:
                tail += stopper.flush()
            if tail:
                job.out.put(("delta", tail))
        self.held = consumed if finish != "cancelled" else []   # the prompt snapshot stays valid
        t2 = time.perf_counter()
        usage = dict(prompt_tokens=len(ids), completion_tokens=gen, total_tokens=len(ids) + gen)
        self.last = dict(prompt_tokens=len(ids), reused_tokens=len(ids) - len(feed), completion_tokens=gen,
                         prefill_s=round(t1 - t0, 3), decode_tok_s=round(gen / max(1e-9, t2 - t1), 2),
                         finish=finish)
        print(f"[mlx_server] prompt {len(ids)} tok ({len(ids) - len(feed)} reused{', ' + how if how else ''}"
              f"{f', anchor at {cut}' if cut and n < cut < len(ids) else ''}) in {t1 - t0:.1f} s | "
              f"{gen} tok in {t2 - t1:.1f} s = {gen / max(1e-9, t2 - t1):.2f} tok/s | {finish} | "
              f"pool hit {self.pool.stats.hit_rate:.0%}", flush=True)
        job.out.put(("done", finish, usage))

    # -- called from HTTP threads -----------------------------------------
    def submit(self, job: Job) -> Job:
        self.jobs.put(job)
        return job


# -------------------------------------------------------------------- http
def create_app(engine: Engine) -> Flask:
    app = Flask(__name__)

    def err(status, msg, kind="invalid_request_error"):
        return jsonify({"error": {"message": msg, "type": kind}}), status

    def text_of(content) -> str:
        if isinstance(content, list):          # OpenAI multi-part content
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return "" if content is None else str(content)

    @app.get("/health")
    def health():
        if engine.load_error:
            return jsonify(status="error", error=engine.load_error), 500
        if not engine.ready.is_set():
            return jsonify(status="loading"), 503
        return jsonify(status="ok", model=engine.model_id, busy=engine.busy,
                       queued=engine.jobs.qsize(), last_request=engine.last)

    @app.get("/v1/models")
    def models():
        return jsonify(object="list", data=[dict(id=engine.model_id, object="model",
                                                 created=int(time.time()), owned_by="local")])

    @app.post("/v1/chat/completions")
    def chat():
        if not engine.ready.wait(timeout=0) or engine.load_error:
            return err(503, engine.load_error or "model is still loading", "server_error")
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return err(400, "'messages' must be a non-empty list")
        if body.get("n") not in (None, 1):
            return err(400, "only n=1 is supported")
        d = engine.defaults
        try:
            stop = body.get("stop") or []
            stop = [stop] if isinstance(stop, str) else [str(s) for s in stop]
            max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
            params = Params(
                max_tokens=int(max_tokens) if max_tokens not in (None, -1) else engine.args.max_context,
                temperature=float(body["temperature"]) if body.get("temperature") is not None else d.temperature,
                top_p=float(body["top_p"]) if body.get("top_p") is not None else d.top_p,
                top_k=int(body["top_k"]) if body.get("top_k") is not None else d.top_k,
                stop=stop, seed=int(body["seed"]) if body.get("seed") not in (None, -1) else None)
        except (TypeError, ValueError) as e:
            return err(400, f"bad sampling parameter: {e}")
        if not (0 <= params.temperature <= 5) or not (0 < params.top_p <= 1) or params.top_k < 0 \
                or params.max_tokens < 1:
            return err(400, "temperature must be in [0, 5], top_p in (0, 1], top_k >= 0, max_tokens >= 1")
        messages = [dict(m, content=text_of(m.get("content"))) for m in body["messages"]]
        tools = body.get("tools") or None
        if tools:
            messages = normalize_messages(messages)    # earlier tool calls: arguments -> dict for the template
        if engine.system_prompt:                       # --system-prompt-file: goes first, every request
            if messages[0].get("role") == "system":
                messages[0] = dict(messages[0], content=engine.system_prompt + "\n\n" + messages[0]["content"])
            else:
                messages.insert(0, {"role": "system", "content": engine.system_prompt})
        anchor = body.get("cache_anchor")
        job = engine.submit(Job(messages, params, tools=tools,
                                anchor=anchor if isinstance(anchor, str) and anchor else None))
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta, finish=None, usage=None):
            c = dict(id=cid, object="chat.completion.chunk", created=created, model=engine.model_id,
                     choices=[dict(index=0, delta=delta, finish_reason=finish)])
            if usage is not None:
                c["usage"] = usage
            return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            def events():
                calls = ToolCallStream(tools) if tools else None   # Qwen3-Coder <tool_call> -> tool_calls
                try:
                    yield chunk({"role": "assistant", "content": ""})
                    while True:
                        try:
                            item = job.out.get(timeout=engine.args.keepalive_s)
                        except queue.Empty:
                            yield ": keepalive\n\n"
                            continue
                        if item[0] == "delta":
                            text = calls.push(item[1]) if calls else item[1]
                            if text:
                                yield chunk({"content": text})
                        elif item[0] == "error":
                            yield f"data: {json.dumps({'error': {'message': item[1], 'type': 'server_error'}})}\n\n"
                            break
                        else:
                            _, finish, usage = item
                            found = []
                            if calls:
                                rest, found = calls.finish()
                                if rest:
                                    yield chunk({"content": rest})
                                for n, tc in enumerate(found):
                                    yield chunk({"tool_calls": [dict(tc, index=n)]})
                            yield chunk({}, "tool_calls" if found else "length" if finish == "length" else "stop")
                            if include_usage:    # OpenAI: a final chunk with empty choices
                                u = dict(id=cid, object="chat.completion.chunk", created=created,
                                         model=engine.model_id, choices=[], usage=usage)
                                yield f"data: {json.dumps(u)}\n\n"
                            break
                    yield "data: [DONE]\n\n"
                finally:
                    job.cancel.set()     # client gone or finished: stop generating

            return Response(stream_with_context(events()), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        parts = []
        while True:
            item = job.out.get()
            if item[0] == "delta":
                parts.append(item[1])
            elif item[0] == "error":
                return err(500, item[1], "server_error")
            else:
                _, finish, usage = item
                break
        text, found = "".join(parts), []
        if tools:
            from tool_calls import parse_tool_calls
            text, found = parse_tool_calls(text, tools)
        message = dict(role="assistant", content=text if (text or not found) else None)
        if found:
            message["tool_calls"] = found
        return jsonify(id=cid, object="chat.completion", created=created, model=engine.model_id,
                       choices=[dict(index=0, message=message,
                                     finish_reason="tool_calls" if found else "length" if finish == "length" else "stop")],
                       usage=usage)

    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL_4BIT), help="MLX checkpoint dir (dense weights, tokenizer)")
    ap.add_argument("--experts", default=str(EXPERTS_4BIT), help="experts.bin/json dir (4-bit or 2-bit)")
    ap.add_argument("--model-id", default=None, help="name reported by /v1/models (default: folder name)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--budget-gb", type=float, default=5.5, help="process memory target")
    ap.add_argument("--reserve-gb", type=float, default=0.9,
                    help="non-pool overhead besides dense weights (KV, state, MLX cache, staging)")
    ap.add_argument("--slots", type=int, default=None, help="expert slots (overrides --budget-gb)")
    ap.add_argument("--spec-slots", type=int, default=64)
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--cache-limit-gb", type=float, default=0.5)
    ap.add_argument("--max-context", type=int, default=16384,
                    help="prompt + completion cap (KV memory grows with it)")
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--keepalive-s", type=float, default=5.0)
    ap.add_argument("--no-prefix-reuse", action="store_true")
    ap.add_argument("--system-prompt-file", default=None,
                    help="text file put at the start of every request's system prompt")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    engine = Engine(args)
    app = create_app(engine)
    print(f"[mlx_server] loading {args.model} + {args.experts}; serving http://{args.host}:{args.port}/v1",
          flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
