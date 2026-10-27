"""
YanahCoder.app entry point: a local coding agent.

Double-clicked, it opens a small control window and:
  1. checks the Mac has at least 16 GB of memory (the 4-bit coder model needs it);
  2. finds the Qwen3-Coder-Next model (or offers the download when this build
     has a model repository; otherwise says how to prepare it);
  3. sizes the engine to the Mac's RAM (--budget-gb, see coder_runtime.BUDGET_TIERS),
     starts mlx_server on port 8888 (the next free port if 8888 is taken) with
     coder_system.md as the system prompt, and restarts it if macOS kills it;
  4. once the engine answers /health, opens the bundled OpenCode desktop app
     (YanahCoder.app/Contents/Resources/OpenCode.app) with `open --env ...`, so it
     is already connected to the engine (no config files to write, no Terminal);
  5. on Quit stops the engine and that OpenCode window.

The same executable runs every part: `YanahCoder --role engine|selftest`.

From source:  python start_coder.py
"""
from __future__ import annotations

import json
import os
import plistlib
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import coder_runtime as crt
from start_yanah import Child, available_memory_gb, http_json, rotate, tail   # shared helpers

PRODUCT = "YanahCoder"


# ============================================================== child roles
def _watch_parent() -> None:
    want = os.environ.get("YANAH_PARENT_PID")
    if not want:
        return
    want = int(want)

    def loop():
        while True:
            time.sleep(2)
            if os.getppid() != want:
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()


def run_role(role: str, args: list) -> int:
    _watch_parent()
    if role == "engine":
        import mlx_server
        mlx_server.main(args)
        return 0
    if role == "selftest":
        return selftest()
    print(f"unknown role {role}", file=sys.stderr)
    return 2


def selftest() -> int:
    """Checks the bundle has everything it needs (build_coder.command runs it)."""
    res, ok = {}, True

    def check(name, fn):
        nonlocal ok
        try:
            res[name] = fn() or "ok"
        except BaseException as e:
            ok = False
            res[name] = f"FAIL {type(e).__name__}: {e}"

    def gpu():
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("Metal not available")
        a = mx.random.normal((256, 256))
        mx.eval((a @ a.T).sum())
        return f"mlx {mx.__version__}, metal ok"

    def lm():
        import mlx_lm
        from mlx_lm.models import qwen3_next  # noqa: F401
        return f"mlx_lm {mlx_lm.__version__}"

    def libs():
        import flask, httpx, huggingface_hub, numpy, transformers, tokenizers, jinja2  # noqa: F401
        return f"transformers {transformers.__version__}"

    def files():
        p = crt.system_prompt_path()
        if not p.is_file():
            raise FileNotFoundError(p)
        return f"{p.name}: {len(p.read_text())} chars"

    def opencode():
        app = crt.opencode_app_path()
        exe = opencode_executable(app)
        if not os.access(exe, os.X_OK):
            raise FileNotFoundError(f"{exe} missing or not executable")
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        where = app.relative_to(app.parents[1]) if crt.FROZEN else app
        if info.get("CFBundleIdentifier") != "app.yanah.coder.app":
            raise RuntimeError(f"OpenCode.app isn't adapted for YanahCoder (bundle id {info.get('CFBundleIdentifier')}); "
                               "build_coder.command runs patch_opencode.py")
        return f"{where}: {info.get('CFBundleName')} {info.get('CFBundleShortVersionString')} ({info.get('CFBundleIdentifier')})"

    def toolcalls():
        from tool_calls import parse_tool_calls
        _, calls = parse_tool_calls("<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n"
                                    "</function>\n</tool_call>")
        assert calls and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}
        return "qwen3_coder parser ok"

    def tokenizer():
        m = crt.find_models()
        if not m:
            return "skipped (no coder model on this Mac)"
        from mlx_lm.tokenizer_utils import load as load_tok
        t = load_tok(Path(m["dense"]))
        return f"{m['dense'].name}: {len(t.encode('def main():'))} tokens, coder template: {crt.is_coder_model(m['dense'])}"

    check("python", lambda: sys.version.split()[0])
    check("gpu", gpu)
    check("mlx_lm", lm)
    check("libraries", libs)
    check("tkinter", lambda: (__import__("_tkinter"), "ok")[1])
    check("modules", lambda: (__import__("mlx_server"), __import__("moe_prefetch.mlx_backend"), "ok")[-1])
    check("system prompt", files)
    check("opencode", opencode)
    check("tool calls", toolcalls)
    check("tokenizer", tokenizer)
    print(json.dumps({"ok": ok, **res}, indent=1))
    return 0 if ok else 1


def child_cmd(role: str, *args) -> list:
    if crt.FROZEN:
        return [sys.executable, "--role", role, *args]
    return [sys.executable, str(Path(__file__).resolve()), "--role", role, *args]


# ============================================================== OpenCode
def opencode_config(port: int, max_context: int) -> dict:
    """OpenCode's whole configuration, passed in OPENCODE_CONFIG_CONTENT (no files)."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "yanah": {
                "npm": "@ai-sdk/openai-compatible",          # bundled inside OpenCode: works offline
                "name": "Yanah Coder (local)",
                "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "local"},
                "models": {crt.MODEL_ID: {
                    "name": "Qwen3-Coder-Next (local)",
                    "tool_call": True,
                    "limit": {"context": max_context, "output": 8192},
                }},
            }
        },
        "model": f"yanah/{crt.MODEL_ID}",
        "small_model": f"yanah/{crt.MODEL_ID}",
        "enabled_providers": ["yanah"],       # only the local engine: no cloud providers, no sign-in
        "autoupdate": False,                  # the bundled version is the tested one
        "share": "disabled",
        # The title helper would send a second, different prompt every turn. The engine keeps one
        # cached conversation, so that would make the next turn re-read everything.
        "agent": {"title": {"disable": True}},
    }


def opencode_home(app: Path | None = None) -> Path:
    """OpenCode's own config/state/database folder inside YanahCoder's, one per OpenCode major
    version: each major version has its own database layout, and an older one refuses a database a
    newer one made ("Database is not empty and has no session table")."""
    app = app or crt.opencode_app_path()
    try:
        ver = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())["CFBundleShortVersionString"]
        major = str(ver).split(".")[0]
    except Exception:
        major = "unknown"
    return crt.data_dir() / f"opencode-v{major}"


def opencode_env(port: int, max_context: int, base: Path | None = None) -> dict:
    base = base or opencode_home()
    return {
        # OpenCode's own config/state/database live in YanahCoder's folder, so a separate
        # OpenCode install (and its background service) never mixes with this one.
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_DATA_HOME": str(base / "data"),
        "XDG_STATE_HOME": str(base / "state"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config(port, max_context)),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",      # don't fetch the cloud model list
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",       # don't pick up ~/.claude prompts/skills
    }


# ---- the OpenCode desktop app (bundled at Contents/Resources/OpenCode.app)
def opencode_executable(app: Path) -> Path:
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    return app / "Contents" / "MacOS" / info["CFBundleExecutable"]


def running_opencode(app: Path) -> tuple[list, list]:
    """Running OpenCode desktop apps: (pids of ours, [(pid, path)] of other copies).
    OpenCode allows one copy at a time: if another one (an OpenCode the user installed)
    is open, ours would hand over to it and quit, and that one isn't connected to Yanah."""
    exe = opencode_executable(app)
    mine = {str(exe), str(exe.resolve())}     # macOS reports some paths unresolved (/var -> /private/var)
    tail = f".app/Contents/MacOS/{exe.name}"
    ours, others = [], []
    try:
        out = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ours, others
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        cmd = cmd.strip()
        if not pid.isdigit() or tail not in cmd or "Helper" in cmd:
            continue
        path = cmd.split(tail, 1)[0] + tail
        if path in mine or str(Path(path).resolve()) in mine:
            ours.append(int(pid))
        else:
            others.append((int(pid), path))
    return ours, others


def stop_pids(pids: list, wait: float = 5.0) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + wait
    while pids and time.time() < end:
        pids = [p for p in pids if _alive(p)]
        time.sleep(0.2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


DEFAULT_WORKSPACE = "~/YanahCoder"      # outside Documents/Desktop: no macOS folder-access prompt
SKIP_WELCOME = {"firstLaunchOnboardingComplete": True}   # OpenCode's own "welcome done" flag


def workspace() -> Path:
    """The folder OpenCode opens in: settings.json "workspace" if set, else ~/YanahCoder (made if missing).
    OpenCode itself remembers the other projects a developer opens."""
    cur = crt.load_settings().get("workspace")               # settings.json can point it elsewhere
    if cur and Path(cur).expanduser().is_dir():
        return Path(cur).expanduser()
    ws = Path(DEFAULT_WORKSPACE).expanduser()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def new_session_link(directory: Path) -> str:
    """OpenCode's own deep link: opens the folder as a project and goes straight to a new chat."""
    from urllib.parse import urlencode
    return "opencode://new-session?" + urlencode({"directory": str(directory)})


def prepare_opencode_app(app: Path) -> None:
    """Before launch: skip OpenCode's first-run welcome, and clear the download flag macOS put on
    the app (the user already approved YanahCoder; without this, macOS asks again for OpenCode)."""
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    app_id = info.get("CFBundleIdentifier", "")
    if app_id and app_id != "ai.opencode.desktop":           # only our own copy's settings, never a user's OpenCode
        store = Path("~/Library/Application Support").expanduser() / app_id / "opencode.settings"
        try:
            cur = json.loads(store.read_text()) if store.exists() else {}
            if not cur.get("firstLaunchOnboardingComplete"):
                store.parent.mkdir(parents=True, exist_ok=True)
                store.write_text(json.dumps({**cur, **SKIP_WELCOME}, indent=2))
        except Exception:
            pass
    try:
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(app)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        pass


def open_command(app: Path, port: int, max_context: int, directory: Path | None = None) -> list:
    """`open` for the bundled OpenCode app, with its settings as environment variables and a link
    that drops straight into a new chat in `directory`.
    -n: always a new process, so it starts with these settings (a running copy would
    keep the ones it started with)."""
    cmd = ["open", "-n"]
    for k, v in opencode_env(port, max_context, opencode_home(app)).items():
        cmd += ["--env", f"{k}={v}"]
    cmd += ["-a", str(app)]
    if directory is not None:
        cmd.append(new_session_link(directory))
    # No keychain: Chromium otherwise stores a key in "<app name> Safe Storage", and macOS asks for the
    # login password whenever a differently signed copy (every YanahCoder update, or a separately
    # installed OpenCode's existing item) reaches for it. This OpenCode only talks to the local engine.
    return cmd + ["--args", "--use-mock-keychain"]


def launch_opencode_app(port: int, max_context: int) -> str | None:
    """Opens the OpenCode window connected to the engine, in a new chat. Returns a message if it can't."""
    app = crt.opencode_app_path()
    if not (app / "Contents" / "Info.plist").is_file():
        return f"OpenCode.app is missing from this build (expected {app})."
    ours, others = running_opencode(app)
    own_id = plistlib.loads((app / "Contents" / "Info.plist").read_bytes()).get("CFBundleIdentifier")
    if others and own_id == "ai.opencode.desktop":
        # an unpatched OpenCode (build without patch_opencode.py) can't run next to another copy
        return ("Another copy of OpenCode is open. Quit it first (OpenCode > Quit OpenCode), then press "
                "Open OpenCode. Only one copy can run at a time, and that one isn't connected to Yanah.")
    stop_pids(ours)                        # ours from earlier: restart it with the current engine settings
    prepare_opencode_app(app)
    started = time.time()
    ws = workspace()
    r = subprocess.run(open_command(app, port, max_context, ws), capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return f"Couldn't open OpenCode: {(r.stderr or r.stdout).strip() or r.returncode}"
    threading.Thread(target=_resend_link_when_ready, args=(app, ws, started), daemon=True).start()
    return None


def _opencode_ready(app: Path, since: float) -> bool:
    """True once OpenCode's log for a run started after `since` says it finished loading."""
    app_id = plistlib.loads((app / "Contents" / "Info.plist").read_bytes()).get("CFBundleIdentifier", "")
    logs = Path("~/Library/Application Support").expanduser() / app_id / "logs"
    try:
        for run in logs.iterdir():
            main = run / "main.log"
            if main.is_file() and main.stat().st_mtime >= since - 1 and "loading task finished" in main.read_text(
                    errors="replace"):
                return True
    except OSError:
        pass
    return False


def _resend_link_when_ready(app: Path, ws: Path, started: float, timeout: float = 120.0) -> None:
    """OpenCode can drop a link that arrives while it is still starting (its window looks the link up
    before it has finished connecting to its own server). Once it has loaded, send the same link to
    the running copy: it then opens the workspace in a new chat. Sending it twice is harmless."""
    end = started + timeout
    while time.time() < end:
        if _opencode_ready(app, started):
            time.sleep(2.0)                                 # let the window finish drawing
            if running_opencode(app)[0]:
                subprocess.run(["open", "-a", str(app), new_session_link(ws)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            return
        time.sleep(0.5)


def stop_opencode_app() -> None:
    app = crt.opencode_app_path()
    if (app / "Contents" / "Info.plist").is_file():
        stop_pids(running_opencode(app)[0])


# ============================================================== engine supervisor
class Supervisor:
    RESTART_WINDOW_S = 600
    MAX_RESTARTS = 3
    BACKOFF_S = (5, 20, 45)

    def __init__(self, post):
        self.post = post
        self.engine = Child("engine", crt.logs_dir() / "engine.log")
        self.stopping = threading.Event()
        self.port = None
        self.models = None
        self.state = "stopped"
        self.ram = crt.total_ram_gb()
        s = crt.load_settings()
        self.budget = float(s.get("budget_gb") or crt.default_budget_gb(self.ram) or 0)
        self.max_context = int(s.get("max_context") or crt.default_max_context(self.ram))
        crt.write_memory_config(self.ram, self.budget, manual=bool(s.get("budget_gb")))
        self._lock = threading.Lock()

    def _args(self) -> list:
        return ["--host", "127.0.0.1", "--port", str(self.port),
                "--model", str(self.models["dense"]), "--experts", str(self.models["experts"]),
                "--model-id", crt.MODEL_ID,
                "--budget-gb", str(self.budget), "--max-context", str(self.max_context),
                "--cache-limit-gb", "0.5",
                "--system-prompt-file", str(crt.system_prompt_path())]

    def _env(self) -> dict:
        env = dict(os.environ)
        env.update(YANAH_PARENT_PID=str(os.getpid()), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        return env

    def start(self, models: dict) -> None:
        self.models = models
        want = int(crt.load_settings().get("engine_port") or 8888)
        self.port = crt.find_free_port(want)
        free = available_memory_gb()
        if free is not None and free < self.budget:
            self.post("warn", text=f"About {free:.1f} GB of memory is free and the engine is set to use "
                                   f"{self.budget:g} GB. Closing other apps will help.")
        self._start_engine()
        threading.Thread(target=self._monitor, daemon=True).start()

    def _start_engine(self) -> None:
        self.state = "loading"
        self.post("engine", state="loading")
        self.engine.start(child_cmd("engine", *self._args()), self._env())

    def restart(self) -> None:
        with self._lock:
            self.engine.crashes.clear()
            self.engine.stop()
            self._start_engine()

    def stop(self) -> None:
        self.stopping.set()
        self.engine.stop()

    def _monitor(self) -> None:
        last = 0.0
        while not self.stopping.wait(1.0):
            with self._lock:
                if self.engine.proc is not None and not self.engine.alive():
                    rc = self.engine.proc.returncode
                    log = tail(self.engine.log)
                    self.engine.proc = None
                    if "Address already in use" in log:
                        self.port = crt.find_free_port(self.port + 1)
                        self._start_engine()
                        continue
                    now = time.time()
                    c = self.engine.crashes
                    while c and now - c[0] > self.RESTART_WINDOW_S:
                        c.popleft()
                    c.append(now)
                    low_mem = rc in (-signal.SIGKILL, 137) or any(
                        w in log.lower() for w in ("out of memory", "insufficient memory", "metal::malloc"))
                    why = ("macOS stopped it because the Mac ran low on memory" if low_mem
                           else f"it stopped unexpectedly (exit code {rc})")
                    if len(c) > self.MAX_RESTARTS:
                        self.state = "failed"
                        self.post("engine", state="failed", text=f"The engine keeps stopping: {why}.",
                                  log=str(self.engine.log))
                        continue
                    wait = self.BACKOFF_S[min(len(c) - 1, len(self.BACKOFF_S) - 1)]
                    self.state = "restarting"
                    self.post("engine", state="restarting", text=f"The engine paused: {why}. Restarting in {wait} s…")
                    self._lock.release()
                    try:
                        if self.stopping.wait(wait):
                            return
                    finally:
                        self._lock.acquire()
                    if self.engine.proc is None:
                        self._start_engine()
                    continue
            if self.state in ("loading", "restarting") and time.time() - last > 1.5:
                last = time.time()
                code, body = http_json(f"http://127.0.0.1:{self.port}/health")
                if code == 200:
                    self.state = "ok"
                    self.post("engine", state="ok")
                elif code == 500:
                    self.state = "failed"
                    self.post("engine", state="failed",
                              text=f"The engine couldn't load the model: {body.get('error', 'unknown error')}",
                              log=str(self.engine.log))
                    self.engine.stop()


# ============================================================== window
class App:
    BG, FG, DIM, ACCENT, ERR = "#15151a", "#ececf1", "#9a9aa6", "#7cc4ff", "#ff8a8a"

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.q: queue.Queue = queue.Queue()
        self.sup = Supervisor(self.post)
        self.dl_cancel = threading.Event()
        self.opened = False

        r = self.root = tk.Tk()
        r.title(PRODUCT)
        r.geometry("560x320")
        r.minsize(480, 280)
        r.configure(bg=self.BG)
        try:
            r.createcommand("::tk::mac::Quit", self.quit)
            r.createcommand("::tk::mac::ReopenApplication", self._reopen)
        except tk.TclError:
            pass
        r.protocol("WM_DELETE_WINDOW", self.quit)
        style = ttk.Style(r)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Accent.TButton", background=self.ACCENT, foreground="#15151a", bordercolor=self.ACCENT,
                        focuscolor=self.ACCENT, padding=(14, 6), font=("Helvetica Neue", 13, "bold"))
        style.configure("C.Horizontal.TProgressbar", troughcolor="#2a2a33", background=self.ACCENT,
                        bordercolor=self.BG, lightcolor=self.ACCENT, darkcolor=self.ACCENT)
        pad = dict(padx=24)
        self.header = tk.Label(r, text="Yanah Coder", font=("Helvetica Neue", 26, "bold"), fg=self.FG, bg=self.BG,
                               anchor="w")
        try:
            self.logo = tk.PhotoImage(file=str(crt.resources_dir() / "yanah_logo.png")).subsample(21)
            self.header.configure(image=self.logo, compound="left", padx=0)
        except Exception:
            self.logo = None
        self.header.pack(fill="x", pady=(20, 4), **pad)
        self.title = tk.Label(r, text="", font=("Helvetica Neue", 14), fg=self.FG, bg=self.BG, anchor="w",
                              justify="left", wraplength=510)
        self.title.pack(fill="x", **pad)
        self.detail = tk.Label(r, text="", font=("Helvetica Neue", 12), fg=self.DIM, bg=self.BG, anchor="w",
                               justify="left", wraplength=510)
        self.detail.pack(fill="x", pady=(6, 0), **pad)
        self.bar = ttk.Progressbar(r, style="C.Horizontal.TProgressbar", maximum=1000)
        self.buttons = tk.Frame(r, bg=self.BG)
        self.buttons.pack(side="bottom", fill="x", pady=18, **pad)
        self.b1 = ttk.Button(self.buttons, text="")
        self.b2 = ttk.Button(self.buttons, text="")
        ttk.Button(self.buttons, text="Quit", command=self.quit).pack(side="right")

    def post(self, kind: str, **info) -> None:
        self.q.put((kind, info))

    def _pump(self) -> None:
        try:
            while True:
                kind, info = self.q.get_nowait()
                self._handle(kind, info)
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    def _show(self, title, detail="", b1=None, b2=None, bar=None, err=False) -> None:
        self.title.configure(text=title, fg=self.ERR if err else self.FG)
        self.detail.configure(text=detail)
        for b, spec in ((self.b1, b1), (self.b2, b2)):
            b.pack_forget()
            if spec:
                b.configure(text=spec[0], command=spec[1])
        if b2:
            self.b2.pack(side="left", padx=(0, 8))
        if b1:
            self.b1.pack(side="left", padx=(0, 8))
        if bar is None:
            self.bar.stop()
            self.bar.pack_forget()
        else:
            self.bar.configure(mode=bar)
            self.bar.pack(fill="x", padx=24, pady=(14, 0))
            if bar == "indeterminate":
                self.bar.start(15)

    # -- states
    def start(self) -> None:
        self.root.after(150, self._pump)
        self.root.after(50, self.check)
        self.root.mainloop()

    def _mem_line(self) -> str:
        return f"Detected {self.sup.ram:.0f} GB Mac. Allocating {self.sup.budget:g} GB for the engine."

    def check(self) -> None:
        if crt.default_budget_gb(self.sup.ram) is None and not crt.load_settings().get("budget_gb"):
            self._show("Yanah Coder needs a Mac with at least 16 GB of memory.",
                       f"This Mac has {self.sup.ram:.0f} GB. The coding model runs its 4-bit experts, which "
                       "need more room than an 8 GB Mac has. (The Yanah chat app does run on 8 GB.)", err=True)
            return
        if not (crt.opencode_app_path() / "Contents" / "Info.plist").is_file():
            self._show("OpenCode is missing from this build.", f"Expected it at {crt.opencode_app_path()}", err=True)
            return
        models = crt.find_models()
        if models:
            self._launch(models)
            return
        repo = crt.model_repo(crt.load_settings())
        if repo:
            self._show("Yanah Coder needs to download its model (about 45 GB).",
                       "The download can be paused and resumes where it left off.",
                       b1=("Start Download", self.download))
        else:
            self._show("The coding model isn't on this Mac yet.",
                       "Run prepare_coder_models.command (it makes ~/Desktop/yanah_coder_models), "
                       "then press Check Again.", b1=("Check Again", self.check), err=True)

    def download(self) -> None:
        import download_models as dm
        repo = crt.model_repo(crt.load_settings())
        self.dl_cancel.clear()
        self._show("Downloading the coding model…", "Starting…", b1=("Pause", self.dl_cancel.set),
                   bar="determinate")

        def progress(p):
            self.post("dl", phase=p.phase, done=p.done_bytes, total=p.total_bytes, rate=p.rate_bps, eta=p.eta_text)

        def work():
            try:
                dm.Downloader(repo, crt.download_root(), "4bit", on_progress=progress, cancel=self.dl_cancel).run()
                self.post("dl_done")
            except dm.Cancelled:
                self.post("dl_error", text="Paused. Your progress is saved.")
            except Exception as e:
                self.post("dl_error", text=dm.friendly(e))
        threading.Thread(target=work, daemon=True).start()

    def _launch(self, models: dict) -> None:
        note = "" if crt.is_coder_model(models["dense"]) else \
            "\nWarning: this model's chat template isn't Qwen3-Coder's; tool calls may not work."
        self._show("Starting the coding engine…", self._mem_line() + "\nLoading Qwen3-Coder-Next (4-bit experts). "
                   "This takes a minute or two." + note, bar="indeterminate")
        try:
            self.sup.start(models)
        except Exception as e:
            self._show("The engine couldn't start.", f"{type(e).__name__}: {e}", err=True,
                       b1=("Try Again", self.check))

    def open_app(self) -> None:
        """Opens (or reopens) the OpenCode window, connected to the engine."""
        problem = launch_opencode_app(self.sup.port, self.sup.max_context)
        self.opened = problem is None
        self._running(problem or "")

    def _running(self, problem: str = "") -> None:
        if problem:
            self._show("The engine is running, but OpenCode didn't open.", problem, err=True,
                       b1=("Open OpenCode", self.open_app))
            return
        self._show("Yanah Coder is running.",
                   f"OpenCode is open in its own window, working in {workspace()}.\n"
                   f"Engine: http://127.0.0.1:{self.sup.port}/v1  ·  {self._mem_line()}\n"
                   "Closing this window stops the engine and OpenCode.",
                   b1=("Open OpenCode", self.open_app))

    def _handle(self, kind: str, i: dict) -> None:
        if kind == "dl":
            if i["total"]:
                self.bar.configure(value=1000 * i["done"] / i["total"])
            speed = f"{i['rate'] / 1e6:.0f} MB/s  ·  " if i["rate"] > 1e5 else ""
            self.detail.configure(text=f"{i['done'] / 1e9:.1f} of {i['total'] / 1e9:.1f} GB  ·  {speed}"
                                       f"{i.get('eta') or 'Calculating...'}")
        elif kind == "dl_done":
            self.check()
        elif kind == "dl_error":
            self._show("The download stopped.", i["text"], err=True, b1=("Resume Download", self.download))
        elif kind == "warn":
            self.detail.configure(text=self.detail.cget("text") + "\n" + i["text"])
        elif kind == "engine":
            st = i["state"]
            if st == "ok":
                if not self.opened:
                    self.open_app()
                else:
                    self._running()
            elif st == "restarting":
                self._show("The engine paused.", i["text"] + "\nClosing other apps helps.", bar="indeterminate")
            elif st == "failed":
                self._show("The engine stopped.", i["text"], err=True, b1=("Restart Engine", self._restart),
                           b2=("Show Log", lambda: subprocess.run(["open", "-R", i.get("log", "")])))

    def _restart(self) -> None:
        self._show("Starting the coding engine…", self._mem_line(), bar="indeterminate")
        threading.Thread(target=self.sup.restart, daemon=True).start()

    def _reopen(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def quit(self) -> None:
        self.dl_cancel.set()
        self._show("Stopping…", bar="indeterminate")
        self.root.update_idletasks()
        try:
            stop_opencode_app()
            self.sup.stop()
        finally:
            self.root.destroy()


# ============================================================== main
def _redirect_output() -> None:
    if crt.FROZEN:
        log = crt.logs_dir() / "app.log"
        rotate(log)
        sys.stdout = sys.stderr = open(log, "a", buffering=1)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--role"] and len(argv) >= 2:
        return run_role(argv[1], argv[2:]) or 0
    _redirect_output()
    App().start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
