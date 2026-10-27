"""
End-to-end test of mlx_server.py on a tiny random Qwen3-Next with a real
byte-level BPE tokenizer and a Qwen-style chat template, over real HTTP.

    python test_mlx_server.py
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time

import httpx
import numpy as np


def build_model(root: str):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    import moe_prefetch.test_mlx_backend as tmb
    from moe_prefetch.repack import repack
    md, ed = os.path.join(root, "model"), os.path.join(root, "experts")
    os.makedirs(md)
    tmb.CFG = dict(tmb.CFG, vocab_size=320)
    tmb.make_checkpoint(md)
    repack(md, ed)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    corpus = ["the quick brown fox jumps over the lazy dog " * 20, "hello world, how are you today? " * 20,
              "system user assistant tools json name arguments " * 20]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=320, special_tokens=specials,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(os.path.join(md, "tokenizer.json"))
    template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    json.dump(dict(tokenizer_class="PreTrainedTokenizerFast", eos_token="<|im_end|>", pad_token="<|endoftext|>",
                   bos_token=None, chat_template=template, model_max_length=4096),
              open(os.path.join(md, "tokenizer_config.json"), "w"))
    ids = {t: tok.token_to_id(t) for t in specials}
    json.dump(dict(eos_token_id=[ids["<|im_end|>"], ids["<|endoftext|>"]], temperature=0.7, top_p=0.8, top_k=20),
              open(os.path.join(md, "generation_config.json"), "w"))
    return md, ed


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def yanah_stream(base, messages, **kw):
    """yanah_shell.py's call_mastermind(stream=True) loop, verbatim logic."""
    payload = {"model": "", "messages": messages, "temperature": 0.7, "max_tokens": 4096, "stream": True, **kw}
    out = []
    with httpx.Client(timeout=300.0, trust_env=False) as c:
        with c.stream("POST", f"{base}/chat/completions", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                tok = json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content")
                if tok:
                    out.append(tok)
    return "".join(out)


def main():
    root = tempfile.mkdtemp()
    md, ed = build_model(root)
    import mlx_server
    from werkzeug.serving import make_server
    port = free_port()
    args = mlx_server.parse_args(["--model", md, "--experts", ed, "--slots", "40", "--spec-slots", "8",
                                  "--port", str(port), "--max-context", "2048", "--keepalive-s", "0.05"])
    engine = mlx_server.Engine(args)
    srv = make_server("127.0.0.1", port, mlx_server.create_app(engine), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"
    h = httpx.Client(timeout=60, trust_env=False)
    for _ in range(600):
        r = h.get(f"http://127.0.0.1:{port}/health")
        if r.status_code == 200:
            break
        time.sleep(0.1)
    assert r.status_code == 200, r.text
    print("[1] /health ok:", r.json()["model"], "| /v1/models:", h.get(f"{base}/models").json()["data"][0]["id"])

    msgs = [{"role": "system", "content": "you are yanah"}, {"role": "user", "content": "hello world"}]
    greedy = dict(model="", messages=msgs, temperature=0, max_tokens=16)
    r1 = h.post(f"{base}/chat/completions", json=greedy).json()
    text = r1["choices"][0]["message"]["content"]
    assert r1["usage"]["completion_tokens"] >= 1 and r1["choices"][0]["finish_reason"] in ("stop", "length")

    print(f"[2] non-stream: {len(text)} chars, usage {r1['usage']}")

    s_text = yanah_stream(base, msgs, temperature=0, max_tokens=16)
    assert s_text == text, (s_text, text)
    print("[3] stream via yanah_shell's parser == non-stream content")

    from openai import OpenAI
    oc = OpenAI(base_url=base, api_key="local", http_client=httpx.Client(trust_env=False, timeout=60))
    parts, usage, finish = [], None, None
    for ch in oc.chat.completions.create(model="x", messages=msgs, temperature=0, max_tokens=16, stream=True,
                                         stream_options={"include_usage": True}):
        if ch.usage:
            usage = ch.usage
        if ch.choices:
            if ch.choices[0].delta.content:
                parts.append(ch.choices[0].delta.content)
            finish = ch.choices[0].finish_reason or finish
    assert "".join(parts) == text and usage and usage.completion_tokens == r1["usage"]["completion_tokens"]
    nc = oc.chat.completions.create(model="x", messages=msgs, temperature=0, max_tokens=16)
    assert nc.choices[0].message.content == text
    print(f"[4] OpenAI client: stream + include_usage + non-stream agree (finish {finish})")

    # prefix reuse: turn 2 extends turn 1 with the assistant reply
    turn2 = msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": "how are you today?"}]
    r2 = h.post(f"{base}/chat/completions", json=dict(model="", messages=turn2, temperature=0, max_tokens=16)).json()
    last = h.get(f"http://127.0.0.1:{port}/health").json()["last_request"]
    assert last["reused_tokens"] > 0, last
    engine.args.no_prefix_reuse = True          # same prompt, rebuilt from scratch
    r2c = h.post(f"{base}/chat/completions", json=dict(model="", messages=turn2, temperature=0, max_tokens=16)).json()
    assert h.get(f"http://127.0.0.1:{port}/health").json()["last_request"]["reused_tokens"] == 0
    engine.args.no_prefix_reuse = False
    assert r2["choices"][0]["message"]["content"] == r2c["choices"][0]["message"]["content"]
    # turn 3 on top of turn 2 (state must have been rewound/continued correctly twice)
    turn3 = turn2 + [{"role": "assistant", "content": r2["choices"][0]["message"]["content"]},
                     {"role": "user", "content": "tell me about the fox"}]
    h.post(f"{base}/chat/completions", json=dict(model="", messages=turn2, temperature=0, max_tokens=16))
    r3a = h.post(f"{base}/chat/completions", json=dict(model="", messages=turn3, temperature=0, max_tokens=16)).json()
    l3 = h.get(f"http://127.0.0.1:{port}/health").json()["last_request"]
    engine.args.no_prefix_reuse = True
    r3c = h.post(f"{base}/chat/completions", json=dict(model="", messages=turn3, temperature=0, max_tokens=16)).json()
    engine.args.no_prefix_reuse = False
    assert l3["reused_tokens"] > 0 and r3a["choices"][0]["message"]["content"] == r3c["choices"][0]["message"]["content"]
    print(f"[5] turn 2 reused {last['reused_tokens']}/{last['prompt_tokens']} prompt tokens, turn 3 "
          f"{l3['reused_tokens']}/{l3['prompt_tokens']}; both replies == a cold rebuild")

    # stop string: cut at the first occurrence of a piece of the greedy text
    if len(text) >= 6:
        stop = text[3:6]
        r3 = h.post(f"{base}/chat/completions", json=dict(greedy, stop=[stop])).json()
        got = r3["choices"][0]["message"]["content"]
        assert got == text[:text.find(stop)] and r3["choices"][0]["finish_reason"] == "stop", (got, text, stop)
        print(f"[6] stop={stop!r}: output cut to {got!r}")

    # client disconnect cancels generation; the next request still works
    with httpx.Client(timeout=60, trust_env=False) as c:
        with c.stream("POST", f"{base}/chat/completions",
                      json=dict(model="", messages=msgs, temperature=0.9, max_tokens=400, stream=True)) as r:
            n = 0
            for line in r.iter_lines():
                if line.startswith("data: "):
                    n += 1
                if n >= 3:
                    break
    for _ in range(200):
        hj = h.get(f"http://127.0.0.1:{port}/health").json()
        if not hj["busy"]:
            break
        time.sleep(0.05)
    assert not hj["busy"] and hj["last_request"]["finish"] == "cancelled", hj
    print(f"[7] disconnect cancelled generation after {hj['last_request']['completion_tokens']} tokens")

    # two concurrent requests serialize and both come back right
    res = [None, None]
    def worker(i):
        res[i] = httpx.post(f"{base}/chat/completions", json=greedy, timeout=120, trust_env=False).json()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(r["choices"][0]["message"]["content"] == text for r in res)
    print("[8] two concurrent requests: both correct (serialized on the engine thread)")

    assert h.post(f"{base}/chat/completions", json={"messages": []}).status_code == 400
    assert h.post(f"{base}/chat/completions", json=dict(greedy, temperature=-1)).status_code == 400
    print("[9] bad requests -> 400")

    # sampled decoding works and respects seed
    a = h.post(f"{base}/chat/completions", json=dict(greedy, temperature=0.8, seed=7)).json()["choices"][0]["message"]["content"]
    b = h.post(f"{base}/chat/completions", json=dict(greedy, temperature=0.8, seed=7)).json()["choices"][0]["message"]["content"]
    assert a == b
    print("[10] temperature 0.8 with a seed is reproducible")
    srv.shutdown()
    print("\nall mlx_server checks passed")


if __name__ == "__main__":
    main()
