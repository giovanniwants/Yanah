"""
Tests for the Yanah.app pieces that don't need a Mac:

  [1] publish -> download round trip: the slim dense file + experts parts,
      downloaded through a fake hub that drops connections and delivers a
      corrupt part, reassemble to byte-identical files
  [2] pause mid-download, resume: only the missing parts are fetched
  [3] verify finds a damaged layer inside experts.bin; repair re-fetches only it
  [4] permanent problems (missing repo, full disk) stop with a plain message
  [5] the slim checkpoint loads in the engine and decodes exactly like the full one
  [6] free-port search skips a port in use
  [7] shell heartbeat: engine down / loading / dies mid-reply / reports OOM
      -> friendly message, conversation rolled back; healthy -> tokens;
      the chat page is served at /
  [8] supervisor: port 8888 taken -> engine on 8889; engine SIGKILLed (what
      macOS does on memory pressure) -> "low on memory" + automatic restart
  [9] children exit when the launcher is gone

    python test_app.py          (needs the moe_prefetch package next to it or one level up)
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if (p / "moe_prefetch").is_dir():
        sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE))

import numpy as np

import download_models as dm
import publish_models as pm
import yanah_runtime as rt


# --------------------------------------------------------------- fixtures
def tiny_models(d: Path):
    from moe_prefetch.repack import repack
    from moe_prefetch.test_mlx_backend import make_checkpoint
    full = d / "Qwen3-80B-4bit"
    full.mkdir(parents=True)
    make_checkpoint(str(full))
    for n in ("tokenizer_config.json", "tokenizer.json"):
        (full / n).write_text("{}")
    exp = d / "Qwen3-80B-2bit-experts"            # (4-bit content; the name is what the app looks for)
    repack(str(full), str(exp))
    return full, exp


def publish(models: Path, out: Path):
    argv = sys.argv
    lic = models / "LICENSE.test"
    lic.write_text("                                 Apache License\n                           Version 2.0, January 2004\n")
    sys.argv = ["publish_models.py", "--dense", str(models / "Qwen3-80B-4bit"),
                "--experts-2bit", str(models / "Qwen3-80B-2bit-experts"), "--out", str(out),
                "--license", str(lic)]
    try:
        pm.main()
    finally:
        sys.argv = argv


class FakeHub:
    """Serves files from a published folder like hf_hub_download(local_dir=...),
    with scripted failures."""

    def __init__(self, remote: Path, drop=(), corrupt=(), missing_repo=False):
        self.remote = remote
        self.drop = {k: n for k, n in drop}           # path -> connection drops before success
        self.corrupt = set(corrupt)                   # path -> delivered damaged once
        self.missing_repo = missing_repo
        self.calls: list = []

    def download(self, filename, local_dir, force=False):
        self.calls.append(filename)
        if self.missing_repo:
            from huggingface_hub.errors import RepositoryNotFoundError
            import httpx
            resp = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/models/x/y"))
            raise RepositoryNotFoundError("404 repo not found", response=resp)
        if self.drop.get(filename, 0) > 0:
            self.drop[filename] -= 1
            import httpx
            raise httpx.ConnectError("network is unreachable")
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.remote / filename, dst)
        if filename in self.corrupt:
            self.corrupt.discard(filename)
            with open(dst, "r+b") as f:
                f.seek(100)
                b = f.read(1)
                f.seek(100)
                f.write(bytes([b[0] ^ 0xFF]))
        return dst

    def remote_sha256(self, paths):
        m = json.loads((self.remote / dm.MANIFEST).read_text())
        allf = m["dense"]["files"] + m["experts"]["2bit"]["parts"] + [m["experts"]["2bit"]["json"]]
        return {f["path"]: f["sha256"] for f in allf if f["path"] in paths}


def same(a: Path, b: Path) -> bool:
    return dm.sha256_file(a) == dm.sha256_file(b)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port):
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --------------------------------------------------------------- tests
def test_download(tmp: Path):
    models = tmp / "src_models"
    full, exp = tiny_models(models)
    remote = tmp / "remote"
    publish(models, remote)
    m = json.loads((remote / dm.MANIFEST).read_text())
    parts = m["experts"]["2bit"]["parts"]
    assert len(parts) >= 4, len(parts)

    # [1] round trip with 2 connection drops on the dense file and a corrupt part
    hub = FakeHub(remote, drop=[("dense/model.safetensors", 2), (parts[1]["path"], 1)],
                  corrupt=[parts[2]["path"]])
    dest = tmp / "dest1"
    events = []
    d = dm.Downloader("x/y", dest, "2bit", hub=hub, sleep=lambda s: None,
                      on_progress=lambda p: events.append((p.phase, p.message)))
    d.run()
    assert same(dest / "Qwen3-80B-2bit-experts" / "experts.bin", exp / "experts.bin")
    assert rt.experts_present(dest / "Qwen3-80B-2bit-experts")
    assert rt.dense_ok(dest / "Qwen3-80B-dense")
    assert any(ph == "waiting" for ph, _ in events), "connection drops should show 'waiting'"
    assert any("damaged" in msg for _, msg in events), "corrupt part should be reported"
    assert hub.calls.count(parts[2]["path"]) == 2, "corrupt part should be fetched exactly twice"
    assert not list((dest / dm.STAGING).glob("experts-2bit/parts/*.bin")), "parts should be deleted after use"
    print(f"[1] download: byte-identical after {sum(1 for ph, _ in events if ph == 'waiting')} retry ticks "
          f"and one corrupt part re-fetched; staging cleaned")

    # the slim dense file has every non-expert tensor and no expert tensors
    from moe_prefetch.repack import read_headers
    hfull, hslim = read_headers(str(full)), read_headers(str(dest / "Qwen3-80B-dense"))
    want = {k for k in hfull if not pm.is_expert(k)}
    assert set(hslim) == want, set(hslim) ^ want

    # [2] pause after 2 parts, then resume
    dest2 = tmp / "dest2"
    cancel = threading.Event()
    hub2 = FakeHub(remote)

    def stop_after_two(p):
        if p.phase == "experts" and hub2.calls.count(parts[1]["path"]):
            cancel.set()
    d = dm.Downloader("x/y", dest2, "2bit", hub=hub2, cancel=cancel, sleep=lambda s: None,
                      on_progress=stop_after_two)
    try:
        d.run()
        raise AssertionError("should have been cancelled")
    except dm.Cancelled:
        pass
    first = [c for c in hub2.calls if "/parts/" in c]
    hub3 = FakeHub(remote)
    dm.Downloader("x/y", dest2, "2bit", hub=hub3, sleep=lambda s: None).run()
    second = [c for c in hub3.calls if "/parts/" in c]
    assert set(second).isdisjoint(set(first[:-1])), "finished parts must not be fetched again"
    assert len(second) <= len(parts) - len(first) + 1
    assert "dense/model.safetensors" not in hub3.calls, "verified dense file must not be fetched again"
    assert same(dest2 / "Qwen3-80B-2bit-experts" / "experts.bin", exp / "experts.bin")
    print(f"[2] pause/resume: {len(first)} parts before pause, {len(second)} after, dense not re-fetched")

    # [3] damage layer 1 inside experts.bin, verify + repair
    eb = dest2 / "Qwen3-80B-2bit-experts" / "experts.bin"
    lb = m["experts"]["2bit"]["layer_bytes"]
    with open(eb, "r+b") as f:
        f.seek(lb + 4096)
        f.write(b"\x00\x01\x02\x03")
    hub4 = FakeHub(remote)
    d = dm.Downloader("x/y", dest2, "2bit", hub=hub4, sleep=lambda s: None)
    bad = d.verify()
    assert bad == [parts[1]["path"]], bad
    d.run()
    fetched = [c for c in hub4.calls if "/parts/" in c]
    assert fetched == [parts[1]["path"]], fetched
    assert same(eb, exp / "experts.bin")
    print("[3] verify: found the damaged layer, repair fetched only that part, file intact")

    # [4] permanent errors
    try:
        dm.Downloader("x/y", tmp / "dest3", "2bit", hub=FakeHub(remote, missing_repo=True),
                      sleep=lambda s: None).run()
        raise AssertionError("expected failure")
    except Exception as e:
        assert dm.classify(e) == "permanent" and "isn't available" in dm.friendly(e), dm.friendly(e)
    real = shutil.disk_usage
    dm.shutil.disk_usage = lambda p: real(p)._replace(free=10 * 2**20)
    try:
        dm.Downloader("x/y", tmp / "dest4", "2bit", hub=FakeHub(remote), sleep=lambda s: None).run()
        raise AssertionError("expected disk-space failure")
    except dm.DownloadError as e:
        assert "disk space" in str(e), e
    finally:
        dm.shutil.disk_usage = real
    print("[4] missing repo and full disk stop with plain messages (no endless retry)")
    return full, exp, dest / "Qwen3-80B-dense"


def test_slim_loads(full: Path, exp: Path, slim: Path):
    from moe_prefetch.mlx_backend import build_runtime
    from moe_prefetch.speculative import MTPSpeculator
    rng = np.random.default_rng(1)
    prompt = [int(t) for t in rng.integers(0, 256, 10)]
    out = []
    for model_dir in (full, slim):
        orch, close = build_runtime(str(model_dir), str(exp), slots=24, sweep_min_tokens=0, spec_lanes=2,
                                    spec_slot_cap=6, mtp_norm_offset=0.0)
        logits, _ = orch.forward(prompt)
        orch.b.reset()
        toks = MTPSpeculator(orch, enabled=False, prefill_chunk=64).generate(prompt, 12)
        close()
        out.append((logits, toks))
    assert np.array_equal(out[0][0], out[1][0]), np.abs(out[0][0] - out[1][0]).max()
    assert out[0][1] == out[1][1]
    print(f"[5] slim checkpoint ({os.path.getsize(slim / 'model.safetensors')} B vs "
          f"{sum(f.stat().st_size for f in full.glob('*.safetensors'))} B full): identical logits and decode")


def test_ports():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not rt.port_is_free(busy)
        p = rt.find_free_port(busy)
        assert p != busy and rt.port_is_free(p)
    print(f"[6] ports: {busy} busy -> picked {p}")


def test_shell(tmp: Path):
    from flask import Flask, Response, jsonify
    eport, sport = free_port(), free_port()
    mode = {"health": "down", "stream": "ok"}
    eng = Flask("fake_engine")

    @eng.get("/health")
    def h():
        return {"ok": (jsonify(status="ok"), 200), "loading": (jsonify(status="loading"), 503),
                "error": (jsonify(status="error", error="experts.json does not match config.json"), 500)}[mode["health"]]

    @eng.post("/v1/chat/completions")
    def chat():
        def gen():
            yield 'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            if mode["stream"] == "die":
                return                                # connection closes without [DONE]
            if mode["stream"] == "oom":
                yield 'data: {"error":{"message":"[metal::malloc] Resource limit exceeded","type":"server_error"}}\n\n'
                return
            yield 'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            yield "data: [DONE]\n\n"
        return Response(gen(), mimetype="text/event-stream")

    data = tmp / "shelldata"
    data.mkdir()
    os.environ.update(YANAH_BASE_URL=f"http://127.0.0.1:{eport}/v1", YANAH_DATA=str(data),
                      YANAH_RESOURCES=str(HERE), YANAH_CONTROL_TOKEN="tok")
    import yanah_shell as ys
    ys.watch.interval = 0.2
    srv = serve(ys.app, sport)
    import httpx
    c = httpx.Client(base_url=f"http://127.0.0.1:{sport}", timeout=20)

    def turn(text):
        evs = []
        with c.stream("POST", "/turn_stream", json={"text": text}) as r:
            ev = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:"):
                    evs.append((ev, json.loads(line[5:])))
        toks = "".join(d["t"] for e, d in evs if e == "token")
        errs = [d["error"] for e, d in evs if e == "error"]
        return toks, errs

    # engine not running at all (nothing listens on eport yet)
    toks, errs = turn("hi")
    assert errs == [ys.MSG_DOWN], errs
    assert ys.daemon.conversation == [], "failed turn must be rolled back"
    es = serve(eng, eport)
    mode["health"] = "loading"
    time.sleep(0.5)
    assert c.get("/health").json()["engine"] == "loading"
    assert turn("hi")[1] == [ys.MSG_LOADING]
    mode["health"] = "error"
    time.sleep(0.5)
    assert "couldn't start" in turn("hi")[1][0]
    mode["health"] = "ok"
    time.sleep(0.5)
    toks, errs = turn("hello")
    assert toks == "Hello" and not errs, (toks, errs)
    assert [m["role"] for m in ys.daemon.conversation] == ["user", "assistant"]
    mode["stream"] = "die"
    toks, errs = turn("again")
    assert toks == "Hel" and errs == [ys.MSG_DOWN], (toks, errs)
    assert len(ys.daemon.conversation) == 2, "half-answered turn must be rolled back"
    mode["stream"] = "oom"
    assert turn("again")[1] == [ys.MSG_DOWN]
    mode["stream"] = "ok"
    time.sleep(0.5)
    assert turn("ok now")[0] == "Hello"
    page = c.get("/")
    assert page.status_code == 200 and "location.origin" in page.text
    assert c.get("/yanah_pact.md").status_code == 404, "identity files must not be served"
    assert c.post("/control/engine", json={"base_url": "http://127.0.0.1:1/v1"}).status_code == 403
    assert (data / "yanah_sessions.json").exists(), "sessions go to the data folder"
    srv.shutdown()
    es.shutdown()
    print("[7] shell: down / loading / load error / died mid-reply / OOM error -> friendly message, "
          "turn rolled back; healthy -> 'Hello'; page served at /, identity files and control guarded")


FAKE_ENGINE = r'''
import os, sys, time, signal, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
marker = os.environ["FAKE_MARKER"]
t0 = time.time()
first_run = not os.path.exists(marker)
open(marker, "a").write("run\n")
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        ok = time.time() - t0 > 0.6
        self.send_response(200 if ok else 503); self.end_headers()
        self.wfile.write(json.dumps({"status": "ok" if ok else "loading"}).encode())
    def log_message(self, *a): pass
srv = ThreadingHTTPServer(("127.0.0.1", port), H)
if first_run:
    import threading
    threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGKILL)).start()   # what jetsam does
print("fake engine on", port, flush=True)
srv.serve_forever()
'''


def test_supervisor(tmp: Path):
    import start_yanah as sy
    fake = tmp / "fake_engine.py"
    fake.write_text(FAKE_ENGINE)
    os.environ["FAKE_MARKER"] = str(tmp / "marker")
    os.environ["YANAH_DATA"] = str(tmp / "supdata")
    real_cmd = sy.child_cmd

    def cmd(role, *args):
        return [sys.executable, str(fake), *args] if role == "engine" else real_cmd(role, *args)
    sy.child_cmd = cmd
    sy.Supervisor.BACKOFF_S = (1, 1, 1)
    base = free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", base))
    blocker.listen()                                  # "something else is using 8888"
    (tmp / "supdata").mkdir(exist_ok=True)
    (tmp / "supdata" / "settings.json").write_text(json.dumps({"engine_port": base, "shell_port": base + 20}))
    posts = []
    sup = sy.Supervisor(lambda kind, **i: posts.append((kind, i)))
    sup.start({"dense": tmp, "experts": tmp, "bits": "2bit"})
    try:
        assert sup.engine_port != base, "busy port must be skipped"
        deadline = time.time() + 30
        while time.time() < deadline:
            states = [i.get("state") for k, i in posts if k == "engine"]
            if states.count("ok") >= 2:
                break
            time.sleep(0.2)
        states = [i.get("state") for k, i in posts if k == "engine"]
        assert states[:3] == ["loading", "ok", "restarting"] and states.count("ok") >= 2, states
        why = next(i["text"] for k, i in posts if k == "engine" and i["state"] == "restarting")
        assert "low on memory" in why, why
        assert any(k == "open_chat" for k, _ in posts), "chat should open once the engine is ready"
        r = sy.http_json(f"http://127.0.0.1:{sup.shell_port}/health")
        assert r[0] == 200
    finally:
        sup.stop()
        blocker.close()
        sy.child_cmd = real_cmd
    assert not sup.engine.alive() and not sup.shell.alive()
    print(f"[8] supervisor: {base} busy -> engine on {sup.engine_port}; SIGKILL -> '{why[:60]}…' -> "
          f"restarted and healthy; stop() leaves nothing running")


def test_orphans(tmp: Path):
    env = dict(os.environ, YANAH_PARENT_PID="1", YANAH_SHELL_PORT=str(free_port()),
               YANAH_DATA=str(tmp / "orph"), YANAH_RESOURCES=str(HERE))
    (tmp / "orph").mkdir(exist_ok=True)
    p = subprocess.Popen([sys.executable, str(HERE / "start_yanah.py"), "--role", "shell"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()
        raise AssertionError("child kept running without its launcher")
    print("[9] a child whose launcher is gone exits on its own")


def main():
    tmp = Path(tempfile.mkdtemp())
    os.environ["YANAH_MODELS"] = str(tmp / "no_models_here")
    full, exp, slim = test_download(tmp)
    test_slim_loads(full, exp, slim)
    test_ports()
    test_shell(tmp)
    test_supervisor(tmp)
    test_orphans(tmp)
    print("all app tests passed")


if __name__ == "__main__":
    main()
