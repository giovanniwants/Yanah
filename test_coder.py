"""
Tests for YanahCoder's own parts (no model or GPU needed; build_coder.command runs them):

  * the Qwen3-Coder tool-call parser (batch and streaming, any token split)
  * mlx_server's HTTP layer with tools: streaming and non-streaming tool_calls,
    finish_reason, --system-prompt-file, and requests without tools unchanged
  * opening the OpenCode desktop app (the `open --env` command, spotting another copy)
  * the RAM tiers

    python test_coder.py
"""
from __future__ import annotations

import json
import os
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
N = 0


def ok(name):
    global N
    N += 1
    print(f"ok {name}")


TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "read", "parameters": {"properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "number"}}}}},
         {"type": "function", "function": {"name": "todowrite", "parameters": {"properties": {
            "todos": {"type": "array"}}}}}]
WRITE = ('Writing it now.\n\n<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.py\n</parameter>\n'
         '<parameter=content>\nprint("<tool_call> inside a string")\n\n</parameter>\n</function>\n</tool_call>')
TWO = ('<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n<parameter=offset>\n10\n</parameter>\n'
       '<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=todowrite>\n'
       '<parameter=todos>\n[{"content": "x", "status": "pending"}]\n</parameter>\n</function>\n</tool_call>')
PROSE = "No tools here, just a < sign and <tool_ fragments and </parameter> text."


def test_parser():
    from tool_calls import ToolCallStream, normalize_messages, parse_tool_calls
    c, calls = parse_tool_calls(WRITE, TOOLS)
    assert c == "Writing it now." and len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/tmp/a.py",
                                                             "content": 'print("<tool_call> inside a string")\n'}
    ok("write call with prose before it; tags inside values kept")
    c, calls = parse_tool_calls(TWO, TOOLS)
    assert [x["function"]["name"] for x in calls] == ["read", "todowrite"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py", "offset": 10, "limit": 5}
    assert json.loads(calls[1]["function"]["arguments"])["todos"][0]["status"] == "pending"
    ok("two calls; integer/number/array arguments typed from the schema")
    assert parse_tool_calls(PROSE, TOOLS) == (PROSE, [])
    ok("plain prose untouched")
    c, calls = parse_tool_calls(TWO.rsplit("</tool_call>", 1)[0], TOOLS)   # stopped before the last close tag
    assert len(calls) == 2
    ok("unclosed last tool call still parsed")
    c, calls = parse_tool_calls("Reading it.\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>", TOOLS)
    assert c == "Reading it." and calls[0]["function"]["name"] == "read"
    ok("call without the <tool_call> wrapper (tokenizer hid it) still parsed")
    for text in (WRITE, TWO, PROSE):
        ref_c, ref = parse_tool_calls(text, TOOLS)
        for _ in range(200):
            s, out, i = ToolCallStream(TOOLS), "", 0
            while i < len(text):
                n = random.randint(1, 9)
                out += s.push(text[i:i + n])
                i += n
            rest, calls = s.finish()
            out += rest
            assert [x["function"] for x in calls] == [x["function"] for x in ref]
            assert out.rstrip("\n") == (ref_c if ref else text).rstrip("\n"), (out, ref_c)
    ok("streaming = batch for 600 random token splits")
    m = normalize_messages([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}}]}])
    assert m[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}
    ok("earlier tool calls: JSON-string arguments -> dict for the chat template")


class FakeEngine:
    """mlx_server's Engine interface, replying with scripted text."""

    def __init__(self, reply, system_prompt=""):
        import mlx_server as ms
        self.reply, self.system_prompt = reply, system_prompt
        self.args = type("A", (), {"keepalive_s": 5.0, "max_context": 4096})()
        self.jobs, self.ready, self.load_error = queue.Queue(), threading.Event(), None
        self.ready.set()
        self.model_id, self.defaults, self.busy, self.last = "test", ms.Params(), False, {}
        self.seen = []

    def submit(self, job):
        self.seen.append(job)

        def run():
            for i in range(0, len(self.reply), 3):
                job.out.put(("delta", self.reply[i:i + 3]))
            job.out.put(("done", "stop", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))
        threading.Thread(target=run, daemon=True).start()
        return job


def sse(resp):
    chunks = []
    for line in resp.get_data(as_text=True).splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def test_server():
    import mlx_server as ms
    eng = FakeEngine(WRITE, system_prompt="CODER PROMPT")
    c = ms.create_app(eng).test_client()
    body = {"messages": [{"role": "system", "content": "harness prompt"}, {"role": "user", "content": "go"}],
            "tools": TOOLS}
    r = c.post("/v1/chat/completions", json=body)
    msg, fin = r.json["choices"][0]["message"], r.json["choices"][0]["finish_reason"]
    assert fin == "tool_calls" and msg["content"] == "Writing it now."
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"])["path"] == "/tmp/a.py"
    ok("non-streaming: message.tool_calls + finish_reason tool_calls")
    assert eng.seen[-1].messages[0]["content"] == "CODER PROMPT\n\nharness prompt"
    ok("--system-prompt-file goes first in the system message")
    r = c.post("/v1/chat/completions", json=dict(body, stream=True))
    ch = sse(r)
    text = "".join(x["choices"][0]["delta"].get("content") or "" for x in ch if x.get("choices"))
    tcs = [tc for x in ch if x.get("choices") for tc in x["choices"][0]["delta"].get("tool_calls") or []]
    fins = [x["choices"][0]["finish_reason"] for x in ch if x.get("choices") and x["choices"][0]["finish_reason"]]
    assert "<tool_call>" not in text and text.strip() == "Writing it now."
    assert len(tcs) == 1 and tcs[0]["index"] == 0 and tcs[0]["function"]["name"] == "write"
    assert fins == ["tool_calls"]
    ok("streaming: no raw <tool_call> text, one tool_calls delta, finish_reason tool_calls")
    eng2 = FakeEngine(WRITE)
    c2 = ms.create_app(eng2).test_client()
    r = c2.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "go"}], "stream": True})
    text = "".join(x["choices"][0]["delta"].get("content") or "" for x in sse(r) if x.get("choices"))
    assert text == WRITE and eng2.seen[-1].messages[0]["role"] == "user"
    ok("no tools in the request (Yanah chat): text passes through unchanged, no system prompt added")
    body = {"messages": [{"role": "user", "content": "go"},
                         {"role": "assistant", "content": None, "tool_calls": [
                             {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a"}'}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "file text"}], "tools": TOOLS}
    eng.reply = "Done."
    r = c.post("/v1/chat/completions", json=body)
    assert r.json["choices"][0]["finish_reason"] == "stop"
    assert eng.seen[-1].messages[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}
    ok("tool results round-trip; prior call arguments decoded for the template")


def fake_opencode_app(root: Path) -> Path:
    """A stand-in OpenCode.app: just the Info.plist and an executable (enough for the launcher)."""
    import plistlib
    app = root / "YanahCoder.app" / "Contents" / "Resources" / "OpenCode.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(
        {"CFBundleExecutable": "OpenCode", "CFBundleIdentifier": "ai.opencode.desktop",
         "CFBundleName": "OpenCode", "CFBundleShortVersionString": "1.19.3"}))
    exe = app / "Contents" / "MacOS" / "OpenCode"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return app


def test_launcher():
    tmp = Path(tempfile.mkdtemp())
    os.environ["YANAH_CODER_DATA"] = str(tmp / "data")
    real = tmp / "real"
    real.mkdir()
    fake_opencode_app(real)
    (tmp / "via_link").symlink_to(real)        # like macOS's /var -> /private/var
    app = tmp / "via_link" / "YanahCoder.app" / "Contents" / "Resources" / "OpenCode.app"
    os.environ["YANAH_OPENCODE_APP"] = str(app)
    import coder_runtime as crt
    import start_coder as sc

    cmd = sc.open_command(app, 8888, 32768)
    assert cmd[:2] == ["open", "-n"] and cmd[-4:] == ["-a", str(app), "--args", "--use-mock-keychain"], cmd
    env = {}
    for i, a in enumerate(cmd):
        if a == "--env":
            k, v = cmd[i + 1].split("=", 1)
            env[k] = v
    ok("launch command: open -n --env ... -a <YanahCoder.app/Contents/Resources/OpenCode.app>")
    cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert cfg["provider"]["yanah"]["options"]["baseURL"] == "http://127.0.0.1:8888/v1"
    assert cfg["provider"]["yanah"]["npm"] == "@ai-sdk/openai-compatible"
    assert cfg["model"] == "yanah/qwen3-coder-next" and cfg["agent"]["title"]["disable"] is True
    assert env["XDG_STATE_HOME"] == str(tmp / "data" / "opencode-v1" / "state")
    ok("OpenCode settings in --env: openai-compatible at 127.0.0.1:8888/v1, own XDG folders, title helper off")
    assert sc.opencode_home(app).name == "opencode-v1"
    ok("OpenCode 1.x gets its own folder (opencode-v1), apart from the 2.x CLI's database")

    exe = sc.opencode_executable(app)
    ps = (f"  101 {exe} --some-flag\n"
          f"  105 {exe.resolve()}\n"
          f"  102 {app}/Contents/Frameworks/OpenCode Helper (Renderer).app/Contents/MacOS/OpenCode Helper (Renderer)\n"
          "  103 /Applications/OpenCode.app/Contents/MacOS/OpenCode\n"
          "  104 /usr/bin/something else\n")
    real_run = sc.subprocess.run
    sc.subprocess.run = lambda *a, **k: type("R", (), {"stdout": ps, "returncode": 0})()
    try:
        ours, others = sc.running_opencode(app)
    finally:
        sc.subprocess.run = real_run
    assert ours == [101, 105] and others == [(103, "/Applications/OpenCode.app/Contents/MacOS/OpenCode")], (ours, others)
    ok("tells our OpenCode apart from another copy (helpers ignored; path shown via a symlink or resolved)")

    os.environ["HOME"] = str(tmp / "home")                     # workspace + OpenCode settings land here
    (tmp / "home").mkdir()
    calls = []
    fake = lambda c, **k: (calls.append(c), type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())[1]
    sc.running_opencode = lambda a: ([], [(103, "/Applications/OpenCode.app/Contents/MacOS/OpenCode")])
    msg = sc.launch_opencode_app(8888, 32768)
    assert msg and "Another copy of OpenCode is open" in msg
    ok("unadapted OpenCode + another copy open: says to quit it instead of opening a disconnected copy")

    import plistlib
    info_p = Path(str(app)) / "Contents" / "Info.plist"
    info = plistlib.loads(info_p.read_bytes())
    info["CFBundleIdentifier"] = "app.yanah.coder.app"          # what patch_opencode.py sets
    info_p.write_bytes(plistlib.dumps(info))
    sc.subprocess.run = fake
    try:
        assert sc.launch_opencode_app(8888, 32768) is None
    finally:
        sc.subprocess.run = real_run
    cmd = next(c for c in calls if c[:2] == ["open", "-n"])
    ok("adapted OpenCode: opens even with another OpenCode running (own app ID)")
    assert cmd[-2:] == ["--args", "--use-mock-keychain"]
    ok("OpenCode starts with --use-mock-keychain (no macOS login-password prompt)")
    link = cmd[-3]
    from urllib.parse import urlparse, parse_qs
    u = urlparse(link)
    assert u.scheme == "opencode" and u.netloc == "new-session", link
    assert Path(parse_qs(u.query)["directory"][0]).is_dir(), link
    ok(f"opens straight into a new chat in the workspace ({link[:60]}...)")
    store = tmp / "home" / "Library" / "Application Support" / "app.yanah.coder.app" / "opencode.settings"
    assert json.loads(store.read_text())["firstLaunchOnboardingComplete"] is True
    ok("OpenCode's first-run welcome is marked done in its own settings (never another OpenCode's)")
    assert any(c[:3] == ["xattr", "-dr", "com.apple.quarantine"] for c in calls)
    ok("clears macOS's download flag on the bundled OpenCode before opening it")

    # the link is sent again once OpenCode's log says it finished loading
    run = tmp / "home" / "Library" / "Application Support" / "app.yanah.coder.app" / "logs" / "20261022T150000"
    run.mkdir(parents=True)
    t0 = time.time()
    assert not sc._opencode_ready(Path(str(app)), t0)
    (run / "main.log").write_text("... server ready\n... loading task finished\n")
    assert sc._opencode_ready(Path(str(app)), t0)
    calls.clear()
    sc.running_opencode = lambda a: ([4242], [])
    sc.time.sleep = lambda s: None
    sc.subprocess.run = fake
    try:
        sc._resend_link_when_ready(Path(str(app)), Path("/tmp/ws"), t0, timeout=5)
    finally:
        sc.subprocess.run = real_run
    assert calls and calls[-1][:3] == ["open", "-a", str(app)] and calls[-1][3].startswith("opencode://new-session?")
    ok("once OpenCode has loaded, the new-chat link is sent again to the running copy")

    tiers = {r: crt.default_budget_gb(r) for r in (8, 16, 18, 24, 32, 36, 48, 64, 128)}
    assert tiers == {8: None, 16: 10.0, 18: 14.0, 24: 18.0, 32: 26.0, 36: 26.0, 48: 40.0, 64: 54.0, 128: 54.0}, tiers
    ok(f"RAM tiers {tiers}")
    sp = Path(__file__).resolve().parent / "coder_system.md"
    assert sp.read_text().startswith("You are an expert software engineer.")
    ok("coder_system.md present")


if __name__ == "__main__":
    test_parser()
    test_server()
    test_launcher()
    print(f"\n{N} checks passed")
