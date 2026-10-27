"""
Speed test: the same ~9,000-token coding prompt through
  1. stock mlx_lm (the whole 4-bit model in memory), and
  2. YanahCoder's engine (mlx_server + moe_prefetch) at several memory budgets.

Measures prompt reading (prefill) and writing (decode) speed for each. Nothing in
YanahCoder is changed; each engine run is a separate process on port 18888.
Results go to build/BENCH_RESULT.txt as each run finishes.

    .venv/bin/python bench_coder.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("YANAH_CODER_MODELS", "~/Desktop/yanah_coder_models")).expanduser()
DENSE = MODELS / "Qwen3-Coder-Next-4bit"
EXPERTS = MODELS / "Qwen3-Coder-Next-4bit-experts"
OUT = HERE / "build" / "BENCH_RESULT.txt"
PORT = 18888
PROMPT_TOKENS = 9000
MAX_NEW = 400

# (label, budget GB, prefill chunk); 256 is what YanahCoder uses today
ENGINE_RUNS = [
    ("engine 54 GB (64 GB Mac), chunk 256 [today]", 54.0, 256),
    ("engine 54 GB (64 GB Mac), chunk 2048", 54.0, 2048),
    ("engine 26 GB (32 GB Mac), chunk 256", 26.0, 256),
    ("engine 10 GB (16 GB Mac), chunk 256", 10.0, 256),
]


def log(line: str) -> None:
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def build_messages(tok) -> list:
    """~PROMPT_TOKENS of real source code plus a coding question."""
    files = ["mlx_server.py", "tool_calls.py", "start_coder.py", "coder_runtime.py",
             "yanah_runtime.py", "moe_prefetch/orchestrator.py", "moe_prefetch/pool.py"]
    parts, used = [], 0
    for name in files:
        p = HERE / name
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        n = len(tok.encode(text))
        if used + n > PROMPT_TOKENS:
            ids = tok.encode(text)[: PROMPT_TOKENS - used]
            text, n = tok.decode(ids), len(ids)
        parts.append(f"### {name}\n```python\n{text}\n```")
        used += n
        if used >= PROMPT_TOKENS:
            break
    user = ("\n\n".join(parts) + "\n\nReview this code. For each file, explain what it does, "
            "then list the three most likely bugs with the exact lines and a fix for each.")
    return [{"role": "system", "content": (HERE / "coder_system.md").read_text().strip()},
            {"role": "user", "content": user}]


def run_stock(messages) -> None:
    code = r'''
import json, sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load(sys.argv[1])
msgs = json.loads(open(sys.argv[2]).read())
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
prompt = list(prompt["input_ids"] if isinstance(prompt, dict) else prompt)
last = None
for r in stream_generate(model, tok, prompt, max_tokens=int(sys.argv[3]), sampler=make_sampler(temp=0.0)):
    last = r
print("RESULT " + json.dumps(dict(prompt_tokens=last.prompt_tokens, prompt_tps=last.prompt_tps,
      gen_tokens=last.generation_tokens, gen_tps=last.generation_tps, peak_gb=last.peak_memory)))
'''
    mfile = HERE / "build" / "bench_messages.json"
    mfile.write_text(json.dumps(messages))
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", code, str(DENSE), str(mfile), str(MAX_NEW)],
                       capture_output=True, text=True, timeout=3600)
    res = next((json.loads(l[7:]) for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not res:
        log(f"stock mlx_lm: FAILED\n{r.stderr[-1500:]}")
        return
    log(f"stock mlx_lm (whole model in memory): prompt {res['prompt_tokens']} tok at "
        f"{res['prompt_tps']:.0f} tok/s ({res['prompt_tokens'] / res['prompt_tps']:.1f} s) | "
        f"write {res['gen_tokens']} tok at {res['gen_tps']:.1f} tok/s | peak {res['peak_gb']:.1f} GB | "
        f"total {time.perf_counter() - t0:.0f} s incl. load")


def get(path: str, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def run_engine(label: str, budget: float, chunk: int, messages) -> None:
    elog = open(HERE / "build" / f"bench_engine_{int(budget)}_{chunk}.log", "w")
    proc = subprocess.Popen([sys.executable, str(HERE / "mlx_server.py"), "--port", str(PORT),
                             "--model", str(DENSE), "--experts", str(EXPERTS),
                             "--budget-gb", str(budget), "--max-context", "32768",
                             "--cache-limit-gb", "0.5", "--prefill-chunk", str(chunk)],
                            stdout=elog, stderr=subprocess.STDOUT, cwd=HERE,
                            env=dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false"))
    try:
        t0 = time.perf_counter()
        while True:
            if proc.poll() is not None:
                log(f"{label}: engine exited while loading (see build/{Path(elog.name).name})")
                return
            try:
                h = get("/health")
                if h.get("status") == "ok":
                    break
            except Exception:
                pass
            if time.perf_counter() - t0 > 1200:
                log(f"{label}: engine didn't load in 20 min")
                return
            time.sleep(2)
        load_s = time.perf_counter() - t0
        body = json.dumps(dict(model="x", messages=messages, max_tokens=MAX_NEW, temperature=0,
                               stream=False)).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.perf_counter()
        try:
            urllib.request.urlopen(req, timeout=3600).read()
        except Exception as e:
            log(f"{label}: request failed after {time.perf_counter() - t1:.0f} s: {e}")
            return
        s = get("/health")["last_request"]
        ptok = s["prompt_tokens"] - s["reused_tokens"]
        log(f"{label}: prompt {ptok} tok at {ptok / max(1e-9, s['prefill_s']):.0f} tok/s "
            f"({s['prefill_s']:.1f} s) | write {s['completion_tokens']} tok at {s['decode_tok_s']:.1f} tok/s "
            f"| load {load_s:.0f} s")
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        elog.close()
        time.sleep(5)


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("")
    for d in (DENSE, EXPERTS):
        if not d.exists():
            log(f"STOPPED: no {d}")
            return
    try:
        get("/health", timeout=2)
        log(f"STOPPED: something is already on port {PORT}")
        return
    except Exception:
        pass
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(DENSE)
    messages = build_messages(tok)
    ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    n = len(ids["input_ids"] if isinstance(ids, dict) else ids)
    log(f"prompt: {n} tokens of real code + a review question; up to {MAX_NEW} tokens written; greedy")
    log(time.strftime("started %H:%M"))
    run_stock(messages)
    for label, budget, chunk in ENGINE_RUNS:
        run_engine(label, budget, chunk, messages)
    log(time.strftime("DONE %H:%M"))


if __name__ == "__main__":
    main()
