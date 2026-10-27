"""
Yanah.app entry point.

Double-clicked, it opens a small control window and:
  1. checks for the models; if they're missing, offers the download
     (resumable, verified: download_models.py) instead of crashing;
  2. picks free ports (engine from 8888 up, chat from 8901 up), so another
     program on 8888/8901 doesn't stop it;
  3. starts the engine (mlx_server) and the chat daemon (yanah_shell) as child
     processes, opens the chat UI in a native pywebview window when the engine is ready;
  4. watches both: if the engine dies (usually macOS killing it for memory),
     the chat shows "Yanah's engine has paused…" and the engine is restarted
     with backoff; if it can't start at all, the window says why;
  5. on Quit (or closing the window) stops both. If this window is killed,
     the children notice and exit, so nothing keeps holding memory.

The same executable runs every part: `Yanah --role engine|shell|download|selftest`.
Inside a PyInstaller app there is no separate python to run scripts with,
so the launcher starts copies of itself with a role.

From source:  python start_yanah.py      (uses ~/Desktop/yanah_models)
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

import yanah_runtime as rt

IDENTITY_FILES = ("personas/default/yanah_self.md", "personas/default/yanah_pact.md", "user_profile.md")


# ============================================================== child roles
def _watch_parent() -> None:
    """Exit if the launcher disappears (force-quit, crash), so an orphaned
    engine never keeps 4+ GB of memory."""
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
    elif role == "shell":
        import yanah_shell
        yanah_shell.main()
    elif role == "download":
        import download_models
        download_models.main(args)
    elif role == "window":
        return run_window(args)
    elif role == "selftest":
        return selftest(args)
    else:
        print(f"unknown role {role}", file=sys.stderr)
        return 2
    return 0


def selftest(args) -> int:
    """Checks the bundle has everything it needs (build_app.command runs it)."""
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
        v = (a @ a.T).sum()
        mx.eval(v)
        return f"mlx {mx.__version__}, metal ok"

    def lm():
        import mlx_lm
        from mlx_lm.models import qwen3_next  # noqa: F401  (imported dynamically by mlx_lm.load)
        return f"mlx_lm {mlx_lm.__version__}"

    def libs():
        import flask, httpx, huggingface_hub, numpy, transformers, tokenizers, jinja2  # noqa: F401
        import flask_cors  # noqa: F401
        return f"transformers {transformers.__version__}, hub {huggingface_hub.__version__}"

    def tk():
        import _tkinter  # noqa: F401
        return "tkinter ok"

    def files():
        r = rt.resources_dir()
        missing = [n for n in ("index.html",) + IDENTITY_FILES if not (r / n).exists()]
        if missing:
            raise FileNotFoundError(f"{missing} not in {r}")
        return str(r)

    def tokenizer():
        m = rt.check_models()
        if not m:
            return "skipped (no models on this Mac)"
        from mlx_lm.tokenizer_utils import load as load_tok
        t = load_tok(Path(m["dense"]))
        return f"{m['dense'].name}: {len(t.encode('hello Yanah'))} tokens"

    check("python", lambda: sys.version.split()[0])
    check("gpu", gpu)
    check("mlx_lm", lm)
    check("libraries", libs)
    check("tkinter", tk)
    check("window", lambda: (__import__("webview"), "pywebview ok")[-1])
    check("yanah modules", lambda: (__import__("mlx_server"), __import__("download_models"),
                                    __import__("moe_prefetch.mlx_backend"), "ok")[-1])
    check("resources", files)
    check("tokenizer", tokenizer)
    print(json.dumps({"ok": ok, **res}, indent=1))
    return 0 if ok else 1


def run_window(args: list, win_holder: dict | None = None) -> int:
    """The chat window: renders index.html in macOS's built-in WebKit via pywebview (no browser needed)."""
    import webview
    url = args[0] if args else "http://127.0.0.1:8901/"
    webview.settings["ALLOW_DOWNLOADS"] = True          # "Export" saves the chat as a .md file
    storage = rt.data_dir() / "webview"                  # keeps the theme and warm-up list between launches
    storage.mkdir(parents=True, exist_ok=True)
    class Api:                                           # window.pywebview.api.* in the page
        win = None

        def set_title(self, title):
            """The page sets the window's title bar (e.g. the idle note)."""
            if Api.win is not None:
                Api.win.set_title(str(title)[:120])

    Api.win = webview.create_window("Yanah", url, width=1280, height=860, min_size=(820, 560),
                                    background_color="#06070a", text_select=True, js_api=Api())
    if win_holder is not None:
        win_holder["win"] = Api.win
    webview.start(private_mode=False, storage_path=str(storage))
    return 0


def run_direct(models: dict) -> int:
    """When models are already installed, skip the Tkinter window completely and launch
    the pywebview chat window directly in the main process (single Dock icon).
    Closing the chat window cleanly shuts down the shell and engine."""
    win_holder: dict = {"win": None}
    initial_url = None

    def post(kind: str, **info) -> None:
        if kind == "open_chat" and win_holder["win"] is not None:
            url = info.get("url")
            if url and url != initial_url:
                try:
                    win_holder["win"].load_url(url)
                except Exception:
                    pass
        elif kind in ("engine", "warn", "info") and info.get("text"):
            print(f"[{kind}] {info['text']}", flush=True)

    sup = Supervisor(post)
    try:
        sup.start(models)
        initial_url = sup.chat_url()
        t_end = time.time() + 30.0
        while time.time() < t_end:
            code, _ = http_json(f"http://127.0.0.1:{sup.shell_port}/health", timeout=1.0)
            if code == 200:
                break
            time.sleep(0.2)
        return run_window([initial_url], win_holder=win_holder)
    finally:
        sup.stop()


def child_cmd(role: str, *args) -> list:
    if rt.FROZEN:
        return [sys.executable, "--role", role, *args]
    return [sys.executable, str(Path(__file__).resolve()), "--role", role, *args]


# ============================================================== helpers
def http_json(url: str, timeout: float = 2.0, data: dict | None = None, headers: dict | None = None):
    import urllib.request
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception:
        return None, {}


def tail(path: Path, n: int = 4000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def rotate(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        os.replace(path, path.with_suffix(path.suffix + ".1"))


def setup_app_directory() -> Path:
    """First launch: ~/Library/Application Support/YanahChat with its models/ subfolder
    (the app keeps its config, chats and model there and never reads ~/Desktop),
    then the default identity files copied into the data folder."""
    base = rt.setup_app_directory()
    seed_identity()
    return base


def seed_identity() -> None:
    """Copy the default identity files to the data folder once, so they can
    be edited (the copies inside the app are read-only)."""
    src, dst = rt.resources_dir(), rt.data_dir()
    if src == dst:
        return
    for n in ("user_profile.md",):
        if (src / n).exists() and not (dst / n).exists():
            shutil.copy2(src / n, dst / n)
    # persona folders: copy any persona/file not there yet; never overwrite your edits
    for f in sorted((src / "personas").rglob("*.md")) if (src / "personas").is_dir() else []:
        out = dst / f.relative_to(src)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)


# ============================================================== supervisor
class Child:
    def __init__(self, name: str, log: Path):
        self.name, self.log = name, log
        self.proc: subprocess.Popen | None = None
        self.crashes: deque = deque()

    def start(self, cmd: list, env: dict) -> None:
        rotate(self.log)
        f = open(self.log, "ab", buffering=0)
        self.proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        f.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, wait: float = 6.0) -> None:
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            p.terminate()
        try:
            p.wait(wait)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                p.kill()
            p.wait(3)


class Supervisor:
    """Starts and watches the engine and the chat daemon. Talks to the UI
    only through `post(kind, **info)` (thread-safe)."""

    RESTART_WINDOW_S = 600
    MAX_RESTARTS = 3
    BACKOFF_S = (5, 20, 45)

    def __init__(self, post):
        self.post = post
        logs = rt.logs_dir()
        prefix = "" if rt.FROZEN else "yanah-app-"
        self.engine = Child("engine", logs / f"{prefix}engine.log")
        self.shell = Child("shell", logs / f"{prefix}shell.log")
        self.stopping = threading.Event()
        self.token = secrets.token_hex(16)
        self.engine_port = self.shell_port = None
        self.models = None
        self.engine_state = "stopped"
        self.opened_chat = False
        self._lock = threading.Lock()
        self._open_lock = threading.Lock()

    # -- config
    def _memory_budget(self) -> float:
        """Detects this Mac's RAM, picks the engine's memory target from the tiers in
        yanah_runtime (settings.json "budget_gb" still wins), writes both to config.json
        and returns the target config.json holds."""
        s = rt.load_settings()
        ram = rt.total_ram_gb()
        free = rt.available_memory_gb()          # psutil: what macOS can hand out right now
        budget = s.get("budget_gb") or rt.default_budget_gb(ram, free)
        rt.write_memory_config(ram, budget, manual=bool(s.get("budget_gb")), available_gb=free)
        print(f"[memory] {ram:.0f} GB Mac, {free if free is None else round(free, 2)} GB free -> "
              f"engine target {budget} GB", flush=True)
        return float(rt.read_memory_config().get("process_target_gb", budget))

    def _engine_args(self) -> list:
        s = rt.load_settings()
        ram = rt.total_ram_gb()
        budget = self._memory_budget()           # from config.json -> --budget-gb (sizes the SlotPool)
        ctx = s.get("max_context") or rt.default_max_context(ram)
        return ["--host", "127.0.0.1", "--port", str(self.engine_port),
                "--model", str(self.models["dense"]), "--experts", str(self.models["experts"]),
                "--budget-gb", str(budget), "--max-context", str(ctx),
                "--cache-limit-gb", str(rt.default_cache_limit_gb(ram))]

    def _env(self) -> dict:
        env = dict(os.environ)
        env.update(YANAH_RESOURCES=str(rt.resources_dir()), YANAH_DATA=str(rt.data_dir()),
                   YANAH_BASE_URL=f"http://127.0.0.1:{self.engine_port}/v1",
                   YANAH_SHELL_PORT=str(self.shell_port), YANAH_CONTROL_TOKEN=self.token,
                   YANAH_PARENT_PID=str(os.getpid()), PYTHONUNBUFFERED="1",
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
        return env

    def chat_url(self) -> str:
        return f"http://127.0.0.1:{self.shell_port}/"

    # -- lifecycle
    def start(self, models: dict) -> None:
        self.models = models
        s = rt.load_settings()
        self.engine_port = rt.find_free_port(int(s.get("engine_port") or 8888))
        self.shell_port = rt.find_free_port(int(s.get("shell_port") or 8901), avoid={self.engine_port})
        seed_identity()
        need = self._memory_budget()
        free = rt.available_memory_gb()
        if free is not None and free < need:
            self.post("warn", text=f"Your Mac has about {free:.1f} GB of memory free and Yanah needs about "
                                   f"{need:.1f} GB. If she struggles, close some apps or browser tabs.")
        self._start_engine()
        self._start_shell()
        threading.Thread(target=self._monitor, daemon=True).start()
        # Open the chat window right away, as soon as the chat service answers (seconds),
        # NOT when the engine has loaded (minutes): the page's cold-boot overlay covers the
        # engine loading and polls /health until it's ready. Waits up to 2 minutes for the
        # service; a slow Mac under memory pressure can take well over 10 s to start it.
        self._launch_t0 = time.time()
        threading.Thread(target=self._wait_shell_then_open, kwargs={"limit_s": 120}, daemon=True).start()

    def _start_engine(self) -> None:
        self.engine_state = "loading"
        self.post("engine", state="loading", port=self.engine_port)
        self.engine.start(child_cmd("engine", *self._engine_args()), self._env())

    def _start_shell(self) -> None:
        self.shell.start(child_cmd("shell"), self._env())

    def restart_engine(self) -> None:
        """Manual restart (button)."""
        with self._lock:
            self.engine.crashes.clear()
            self.engine.stop()
            self._start_engine()

    def stop(self) -> None:
        self.stopping.set()
        self.shell.stop()
        self.engine.stop()

    def _tell_shell_engine_url(self) -> None:
        http_json(f"http://127.0.0.1:{self.shell_port}/control/engine",
                  data={"base_url": f"http://127.0.0.1:{self.engine_port}/v1"},
                  headers={"X-Yanah-Token": self.token}, timeout=3)

    def _monitor(self) -> None:
        last_health = 0.0
        while not self.stopping.wait(1.0):
            with self._lock:
                # chat daemon: cheap to restart
                if self.shell.proc is not None and not self.shell.alive():
                    log = tail(self.shell.log)
                    if "Address already in use" in log:
                        self.shell_port = rt.find_free_port(self.shell_port + 1, avoid={self.engine_port})
                    self.post("info", text="Restarting the chat service…")
                    old_port = self.shell_port if "Address already in use" not in log else None
                    self._start_shell()
                    if old_port is None and self.engine_state == "ok":
                        self._wait_shell_then_open()   # new port: the open page is dead, open the new one

                # engine process died
                if self.engine.proc is not None and not self.engine.alive():
                    rc = self.engine.proc.returncode
                    log = tail(self.engine.log)
                    self.engine.proc = None
                    if "Address already in use" in log:
                        self.engine_port = rt.find_free_port(self.engine_port + 1, avoid={self.shell_port})
                        self._start_engine()
                        self._tell_shell_engine_url()
                        continue
                    now = time.time()
                    crashes = self.engine.crashes
                    while crashes and now - crashes[0] > self.RESTART_WINDOW_S:
                        crashes.popleft()
                    crashes.append(now)
                    low_mem = rc in (-signal.SIGKILL, 137) or any(
                        w in log.lower() for w in ("out of memory", "insufficient memory", "metal::malloc", "kiogpucommandbuffercallbackerroroutofmemory"))
                    why = ("macOS stopped it because the Mac ran low on memory" if low_mem
                           else f"it stopped unexpectedly (exit code {rc})")
                    if len(crashes) > self.MAX_RESTARTS:
                        self.engine_state = "failed"
                        self.post("engine", state="failed", text=f"Yanah's engine keeps stopping: {why}. "
                                  "Close some apps, then press Restart Engine.", log=str(self.engine.log))
                        continue
                    wait = self.BACKOFF_S[min(len(crashes) - 1, len(self.BACKOFF_S) - 1)]
                    self.engine_state = "restarting"
                    self.post("engine", state="restarting",
                              text=f"Yanah's engine paused: {why}. Restarting in {wait} s…")
                    self._lock.release()
                    try:
                        if self.stopping.wait(wait):
                            return
                    finally:
                        self._lock.acquire()
                    if self.engine.proc is None:
                        self._start_engine()
                    continue

            # engine alive: follow its loading state
            if self.engine_state in ("loading", "restarting") and time.time() - last_health > 1.5:
                last_health = time.time()
                code, body = http_json(f"http://127.0.0.1:{self.engine_port}/health")
                if code == 200:
                    self.engine_state = "ok"
                    self.post("engine", state="ok")
                    if not self.opened_chat:
                        self._wait_shell_then_open()
                elif code == 500:
                    # loaded and failed: bad/missing model files, not memory. Don't loop.
                    self.engine_state = "failed"
                    err = body.get("error", "unknown error")
                    self.post("engine", state="failed", text=f"Yanah's engine couldn't load her model: {err}",
                              log=str(self.engine.log), verify=True)
                    self.engine.stop()

    def _wait_shell_then_open(self, limit_s: float = 10.0) -> None:
        t_end = time.time() + limit_s
        while not self.stopping.is_set() and time.time() < t_end:
            if self.opened_chat:
                return
            code, body = http_json(f"http://127.0.0.1:{self.shell_port}/health", timeout=1)
            if code == 200:
                with self._open_lock:                 # the early opener and the monitor can race
                    if self.opened_chat:
                        return
                    self.opened_chat = True
                t0 = getattr(self, "_launch_t0", None)
                print(f"[launch] opening the chat window"
                      f"{f' {time.time() - t0:.1f} s after launch' if t0 else ''}; "
                      f"engine: {(body or {}).get('engine', '?')}", flush=True)
                self.post("open_chat", url=self.chat_url())
                return
            time.sleep(0.25)
        if not self.opened_chat:
            print(f"[launch] the chat service didn't answer within {limit_s:.0f} s; "
                  "the window opens when the engine is ready", flush=True)


# ============================================================== window
def render_markdown(text, md: str, fg: str, dim: str, accent: str) -> None:
    """Draw a small Markdown subset into a read-only tk.Text: # / ## headings,
    paragraphs, numbered and bulleted lists, **bold** and *italic*."""
    import re
    import tkinter.font as tkfont
    base = tkfont.Font(family="Helvetica Neue", size=14)
    fam = base.actual("family")
    text.configure(state="normal")
    text.delete("1.0", "end")
    text.tag_configure("h1", font=(fam, 24, "bold"), foreground=fg, spacing3=6)
    text.tag_configure("h2", font=(fam, 15, "bold"), foreground=accent, spacing1=14, spacing3=3)
    text.tag_configure("p", font=(fam, 13), foreground=fg, spacing3=8)
    text.tag_configure("lead", font=(fam, 14), foreground=dim, spacing3=2)
    text.tag_configure("li", font=(fam, 13), foreground=fg, lmargin1=4, lmargin2=24, spacing3=6)
    text.tag_configure("b", font=(fam, 13, "bold"))
    text.tag_configure("i", font=(fam, 13, "italic"))
    inline = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")

    def emit(s: str, block: str) -> None:
        for piece in inline.split(s):
            if not piece:
                continue
            if piece.startswith("**") and piece.endswith("**"):
                text.insert("end", piece[2:-2], (block, "b"))
            elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
                text.insert("end", piece[1:-1], (block, "i"))
            else:
                text.insert("end", piece, block)
        text.insert("end", "\n", block)

    after_h1 = False
    for raw in md.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("# "):
            emit(s[2:], "h1")
            after_h1 = True
            continue
        if s.startswith("## "):
            emit(s[3:], "h2")
        elif re.match(r"^\d+\.\s", s):
            n, rest = s.split(".", 1)
            emit(f"{n}.  {rest.strip()}", "li")
        elif s[:2] in ("- ", "* "):
            emit("•  " + s[2:], "li")
        else:
            emit(s, "lead" if after_h1 else "p")
        after_h1 = False
    text.configure(state="disabled")


class App:
    BG, FG, DIM, ACCENT, ERR = "#15151a", "#ececf1", "#9a9aa6", "#b99cff", "#ff8a8a"

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.q: queue.Queue = queue.Queue()
        self.sup = Supervisor(self.post)
        self.dl_thread = None
        self.dl_cancel = threading.Event()
        self.mode = None

        r = self.root = tk.Tk()
        r.title("Yanah")
        r.geometry("520x300")
        r.minsize(460, 260)
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
        style.configure("Accent.TButton", background=self.ACCENT, foreground="#15151a",
                        bordercolor=self.ACCENT, focuscolor=self.ACCENT, padding=(14, 6),
                        font=("Helvetica Neue", 13, "bold"))
        style.map("Accent.TButton", background=[("active", "#cbb6ff"), ("pressed", "#a385f5")])
        style.configure("Y.Horizontal.TProgressbar", troughcolor="#2a2a33", background=self.ACCENT,
                        bordercolor=self.BG, lightcolor=self.ACCENT, darkcolor=self.ACCENT)

        pad = dict(padx=24)
        self.header = tk.Label(r, text="Yanah", font=("Helvetica Neue", 26, "bold"), fg=self.FG, bg=self.BG,
                               anchor="w")
        try:   # the YN monogram beside the name (1024 px PNG -> 48 px)
            self.logo_img = tk.PhotoImage(file=str(rt.resources_dir() / "yanah_logo.png")).subsample(21)
            self.header.configure(image=self.logo_img, compound="left", padx=0)
        except Exception:
            self.logo_img = None
        self.header.pack(fill="x", pady=(20, 4), **pad)
        # first-run welcome page (welcome.md), shown instead of header/title/detail
        self.welcome = tk.Text(r, bg=self.BG, fg=self.FG, relief="flat", borderwidth=0, highlightthickness=0,
                               wrap="word", cursor="arrow", padx=0, pady=0, spacing2=3,
                               font=("Helvetica Neue", 14))
        self.welcome_scroll = ttk.Scrollbar(r, orient="vertical", command=self.welcome.yview)
        self.welcome.configure(yscrollcommand=self.welcome_scroll.set)
        self.welcome_shown = False
        self.title = tk.Label(r, text="", font=("Helvetica Neue", 14), fg=self.FG, bg=self.BG,
                              anchor="w", justify="left", wraplength=470)
        self.title.pack(fill="x", **pad)
        self.detail = tk.Label(r, text="", font=("Helvetica Neue", 12), fg=self.DIM, bg=self.BG,
                               anchor="w", justify="left", wraplength=470)
        self.detail.pack(fill="x", pady=(6, 0), **pad)
        self.bar = ttk.Progressbar(r, style="Y.Horizontal.TProgressbar", maximum=1000)
        self.buttons = tk.Frame(r, bg=self.BG)
        self.buttons.pack(side="bottom", fill="x", pady=18, **pad)
        self.btn_primary = ttk.Button(self.buttons, text="")
        self.btn_second = ttk.Button(self.buttons, text="")
        self.btn_quit = ttk.Button(self.buttons, text="Quit", command=self.quit)
        self.btn_quit.pack(side="right")

    # -- thread-safe posting
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

    # -- layout helpers
    def _show(self, title: str, detail: str = "", primary=None, second=None, bar: str | None = None,
              err: bool = False) -> None:
        if self.welcome_shown:                     # back to the compact status layout
            self.welcome_shown = False
            self.welcome.pack_forget()
            self.welcome_scroll.pack_forget()
            self.btn_primary.configure(style="TButton")
            self.header.pack(fill="x", pady=(20, 4), padx=24)
            self.title.pack(fill="x", padx=24)
            self.detail.pack(fill="x", pady=(6, 0), padx=24)
            self.root.geometry("520x300")
        self.title.configure(text=title, fg=self.ERR if err else self.FG)
        self.detail.configure(text=detail)
        for b, spec in ((self.btn_primary, primary), (self.btn_second, second)):
            b.pack_forget()
            if spec:
                b.configure(text=spec[0], command=spec[1])
        if second:
            self.btn_second.pack(side="left", padx=(0, 8))
        if primary:
            self.btn_primary.pack(side="left", padx=(0, 8))
        if bar is None:
            self.bar.stop()
            self.bar.pack_forget()
        else:
            self.bar.configure(mode=bar)
            self.bar.pack(fill="x", padx=24, pady=(14, 0))
            if bar == "indeterminate":
                self.bar.start(15)
            else:
                self.bar.stop()

    # -- states
    def start(self) -> None:
        self.root.after(150, self._pump)
        self.root.after(50, self.check_models)
        self.root.mainloop()

    def check_models(self) -> None:
        models = rt.check_models()
        if models:
            self._launch(models)
            return
        want = rt.wanted_download()
        size = "about 30 GB" if want == "2bit" else "about 45 GB"
        self.mode = "need_download"
        self._show_welcome(size)

    def _show_welcome(self, size: str) -> None:
        """The first-run page: welcome.md, rendered, with its [ Button ] line as the real button."""
        self._show("", primary=None)
        try:
            md = (rt.resources_dir() / "welcome.md").read_text(encoding="utf-8")
        except OSError:
            md = ("# Welcome to Yanah.\n\nYanah needs to download her brain ({download_size}).\n\n"
                  "[ Start Installation ]\n")
        md = md.replace("{download_size}", size)
        label = "Start Installation"
        lines = []
        for line in md.splitlines():
            s = line.strip()
            if s.startswith("[") and s.endswith("]") and "(" not in s:
                label = s.strip("[] ").strip() or label     # the button line
            else:
                lines.append(line)
        for w in (self.header, self.title, self.detail):
            w.pack_forget()
        render_markdown(self.welcome, "\n".join(lines), fg=self.FG, dim=self.DIM, accent=self.ACCENT)
        self.root.geometry("660x780")
        self.welcome.pack(side="left", fill="both", expand=True, padx=(32, 26), pady=(24, 0))
        self.welcome_shown = True
        self.btn_primary.configure(text=label, command=self.start_download, style="Accent.TButton")
        self.btn_primary.pack(side="left", padx=(0, 8))
        self.root.update_idletasks()
        if self.welcome.yview() != (0.0, 1.0):       # only when the text doesn't fit (small screens)
            self.welcome_scroll.pack(side="right", fill="y", pady=(24, 0), padx=(0, 6))

    def start_download(self, verify: bool = False) -> None:
        import download_models as dm
        repo = rt.model_repo(rt.load_settings())
        if repo.startswith(("SET-ME", "your-hf-username")):
            self._show("Yanah's download source isn't set up in this build.",
                       "Build the app with MODEL_REPO=yourname/repo (see BUILDING.md), or put the models in "
                       f"{rt.download_root()}.", err=True, primary=("Check Again", self.check_models))
            return
        self.dl_cancel.clear()
        self.mode = "downloading"
        self._show("Downloading Yanah's brain…", "Starting…",
                   primary=("Pause", self.pause_download), bar="determinate")

        def progress(p):
            self.post("dl", phase=p.phase, done=p.done_bytes, total=p.total_bytes,
                      rate=p.rate_bps, eta=p.eta_text, message=p.message)

        def work():
            d = dm.Downloader(repo, rt.download_root(), rt.wanted_download(), on_progress=progress,
                              cancel=self.dl_cancel)
            try:
                if verify:
                    d.verify()
                d.run()
                self.post("dl_done")
            except dm.Cancelled:
                self.post("dl_paused")
            except Exception as e:
                self.post("dl_error", text=dm.friendly(e))
        self.dl_thread = threading.Thread(target=work, daemon=True)
        self.dl_thread.start()

    def pause_download(self) -> None:
        self.dl_cancel.set()
        self._show("Pausing…", "Finishing the current step.", bar="indeterminate")

    def _launch(self, models: dict) -> None:
        self.mode = "starting"
        bits = "2-bit" if models["bits"] == "2bit" else "4-bit"
        self._show("Waking Yanah up…", f"Loading her model ({bits} experts). This takes a minute or two.",
                   bar="indeterminate")
        try:
            self.sup.start(models)
        except Exception as e:
            self._show("Yanah couldn't start.", f"{type(e).__name__}: {e}", err=True,
                       primary=("Try Again", self.check_models))

    def _reopen(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def open_chat(self, url: str | None = None) -> None:
        """Open the chat in Yanah's own window (pywebview: macOS's built-in WebKit), not in a
        browser. One window at a time; a new address (the chat service moved port) replaces it.
        If the window can't start, the default browser opens the chat instead."""
        url = url or self.sup.chat_url()
        win = getattr(self, "window_proc", None)
        if win is not None and win.poll() is None:
            if getattr(self, "window_url", None) == url:
                return                                    # already open
            win.terminate()
        env = dict(os.environ, YANAH_PARENT_PID=str(os.getpid()), YANAH_DATA=str(rt.data_dir()))
        try:
            self.window_proc = subprocess.Popen(child_cmd("window", url), env=env,
                                                stdout=sys.stdout, stderr=subprocess.STDOUT)
            self.window_url = url
        except OSError as e:
            print(f"[window] couldn't open Yanah's window ({e}); using the default browser", flush=True)
            webbrowser.open(url)

    def _handle(self, kind: str, i: dict) -> None:
        if kind == "dl" and self.mode == "downloading":
            if i["total"]:
                self.bar.configure(value=1000 * i["done"] / i["total"])
            line = f"{i['done'] / 1e9:.1f} of {i['total'] / 1e9:.1f} GB"
            if i["phase"] != "verifying":
                # smoothed in download_models.py (10 s window + EMA); always shows something
                speed = f"{i['rate'] / 1e6:.0f} MB/s  ·  " if i["rate"] > 1e5 else ""
                line += f"  ·  {speed}{i.get('eta') or 'Calculating...'}"
            self.detail.configure(text=line + ("\n" + i["message"] if i["message"] else ""))
            self.title.configure(text="Waiting for the internet…" if i["phase"] == "waiting"
                                 else "Checking Yanah's files…" if i["phase"] == "verifying"
                                 else "Downloading Yanah's brain…", fg=self.FG)
        elif kind == "dl_done":
            self.check_models()
        elif kind == "dl_paused":
            self.mode = "need_download"
            self._show("Download paused.", "Your progress is saved.",
                       primary=("Resume Download", self.start_download))
        elif kind == "dl_error":
            self.mode = "need_download"
            self._show("The download stopped.", i["text"], err=True,
                       primary=("Try Again", self.start_download),
                       second=("Check Files", lambda: self.start_download(verify=True)))
        elif kind == "warn":
            self.detail.configure(text=i["text"])
        elif kind == "info":
            self.detail.configure(text=i["text"])
        elif kind == "engine":
            st = i["state"]
            if st == "loading" and self.mode != "running":
                self.mode = "starting"
            elif st == "ok":
                self.mode = "running"
                self._show("Yanah is running.", "Closing this window stops Yanah.",
                           primary=("Open Chat", self.open_chat))
            elif st == "restarting":
                self.mode = "starting"
                self._show("Yanah's engine paused.", i["text"] + "\nClosing some apps or browser tabs helps.",
                           bar="indeterminate")
            elif st == "failed":
                self.mode = "failed"
                second = ("Check Files", lambda: self.start_download(verify=True)) if i.get("verify") else \
                    ("Show Log", lambda: subprocess.run(["open", "-R", i.get("log", "")]))
                self._show("Yanah's engine stopped.", i["text"], err=True,
                           primary=("Restart Engine", self._restart_engine), second=second)
        elif kind == "open_chat":
            self.open_chat(i["url"])

    def _restart_engine(self) -> None:
        self._show("Waking Yanah up…", "Loading her model. This takes a minute or two.", bar="indeterminate")
        threading.Thread(target=self.sup.restart_engine, daemon=True).start()

    def quit(self) -> None:
        self.dl_cancel.set()
        self._show("Stopping Yanah…", bar="indeterminate")
        self.root.update_idletasks()
        try:
            win = getattr(self, "window_proc", None)
            if win is not None and win.poll() is None:
                win.terminate()                           # close the chat window with the app
            self.sup.stop()
        finally:
            self.root.destroy()


# ============================================================== main
def _redirect_output() -> None:
    """A Finder-launched app has no terminal; keep a log instead."""
    if rt.FROZEN:
        log = rt.logs_dir() / "app.log"
        rotate(log)
        f = open(log, "a", buffering=1)
        sys.stdout = sys.stderr = f


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--role"] and len(argv) >= 2:
        return run_role(argv[1], argv[2:]) or 0
    # Finder/LaunchServices may pass -psn_… on older macOS: ignore it
    if rt.FROZEN:
        rt.setup_app_directory()            # before the log file goes into its logs/ folder
    _redirect_output()
    os.environ.setdefault("YANAH_DATA", str(rt.data_dir()))
    if rt.FROZEN:
        setup_app_directory()
    models = rt.check_models()
    if models:
        return run_direct(models)
    App().start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
