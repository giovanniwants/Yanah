"""
Shared runtime settings for the Yanah app: where things live, how much
memory to use, and free ports. Used by start_yanah.py (the launcher),
download_models.py and yanah_shell.py.

Where things live
-----------------
resources   read-only code + index.html + default identity files.
            In Yanah.app that is Contents/Frameworks (PyInstaller's
            sys._MEIPASS); from source it is the yanah_build folder.
data        writable: chats (yanah_sessions.json), memories, editable identity
            files, settings, config.json, logs. In the app:
            ~/Library/Application Support/YanahChat, made by setup_app_directory()
            with its models/ subfolder.
            From source: the yanah_build folder (unchanged behaviour).
models      In the app: $YANAH_MODELS, else ~/Library/Application Support/YanahChat/models
            only. The app never looks in ~/Desktop or ~/Documents: reading those
            makes macOS ask the user for access to their files.
            From source: ~/Desktop/yanah_models first (your existing folder), then
            Application Support.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

FROZEN = bool(getattr(sys, "frozen", False))
APP_SUPPORT = Path("~/Library/Application Support").expanduser() / "YanahChat"
OLD_APP_SUPPORT = Path("~/Library/Application Support").expanduser() / "Yanah"   # before 2026-10-23
DESKTOP_MODELS = Path("~/Desktop/yanah_models").expanduser()     # source runs only, never the app
APP_SUBDIRS = ("models",)

DENSE_DIRS = ("Qwen3-80B-4bit", "Qwen3-80B-dense")     # full 4-bit checkpoint, or the slim download
EXPERT_DIRS = {"2bit": "Qwen3-80B-2bit-experts", "4bit": "Qwen3-80B-4bit-experts"}

# Filled in by build_app.command from MODEL_REPO; the env var wins.
DEFAULT_MODEL_REPO = "Giovanniwants/Yanah-Qwen3-Next-80B"

DEFAULT_SETTINGS = {
    "model_repo": None,          # None -> build config / env / DEFAULT_MODEL_REPO
    "experts": "auto",           # auto | 2bit | 4bit
    "budget_gb": None,           # None -> by RAM (default_budget_gb)
    "max_context": None,         # None -> by RAM
    "engine_port": 8888,
    "shell_port": 8901,
}


# ------------------------------------------------------------------ places
def resources_dir() -> Path:
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def _merge_into(old: Path, new: Path) -> None:
    """Moves everything from old into new that new doesn't have yet (folders merged one
    level at a time); never overwrites. Removes old when it ends up empty."""
    for item in list(old.iterdir()):
        target = new / item.name
        try:
            if not target.exists():
                item.rename(target)
            elif item.is_dir() and target.is_dir():
                _merge_into(item, target)
        except OSError:
            pass
    try:
        old.rmdir()                     # only succeeds when empty
    except OSError:
        pass


def setup_app_directory(root=None) -> Path:
    """Makes ~/Library/Application Support/YanahChat and its models/ subfolder if they're
    missing; safe to call on every launch. An earlier install's folder
    (~/Library/Application Support/Yanah: chats, settings, a downloaded model) is renamed
    to YanahChat the first time, so nothing is lost or downloaded again."""
    base = Path(root or APP_SUPPORT).expanduser()
    if root is None and OLD_APP_SUPPORT.is_dir():
        if not base.exists():
            OLD_APP_SUPPORT.rename(base)
        else:            # YanahChat already made (e.g. the model moved in by hand): bring the rest over
            _merge_into(OLD_APP_SUPPORT, base)
    base.mkdir(parents=True, exist_ok=True)
    for sub in APP_SUBDIRS:
        (base / sub).mkdir(exist_ok=True)
    return base


def data_dir() -> Path:
    env = os.environ.get("YANAH_DATA")
    if env:
        d = Path(env).expanduser()
    elif FROZEN:
        return setup_app_directory()
    else:
        d = resources_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = (setup_app_directory() / "logs") if FROZEN else Path("~").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_roots() -> list:
    env = os.environ.get("YANAH_MODELS")
    if env:
        return [Path(env).expanduser()]
    if FROZEN:
        return [APP_SUPPORT / "models"]         # the app never looks in ~/Desktop
    return [DESKTOP_MODELS, APP_SUPPORT / "models"]


def download_root() -> Path:
    env = os.environ.get("YANAH_MODELS")
    return Path(env).expanduser() if env else APP_SUPPORT / "models"


# ---------------------------------------------------------------- settings
def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    p = data_dir() / "settings.json"
    try:
        s.update(json.loads(p.read_text()))
    except FileNotFoundError:
        pass
    except Exception as e:                      # a broken file must not stop the app
        print(f"[yanah] ignoring unreadable {p}: {e}", file=sys.stderr)
    return s


def model_repo(settings: Optional[dict] = None) -> str:
    if os.environ.get("YANAH_MODEL_REPO"):
        return os.environ["YANAH_MODEL_REPO"]
    if settings and settings.get("model_repo"):
        return settings["model_repo"]
    try:
        cfg = json.loads((resources_dir() / "yanah_build_config.json").read_text())
        if cfg.get("model_repo"):
            return cfg["model_repo"]
    except Exception:
        pass
    return DEFAULT_MODEL_REPO


# ------------------------------------------------------------------ memory
def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 8.0


# Engine process memory target (process_target_gb) by the Mac's physical RAM. The rest is
# left for macOS and other apps. Sizes between tiers (36 GB, 48 GB...) use the tier below.
BUDGET_TIERS = (     # (RAM GB, process_target_gb)
    (64, 48.0),
    (32, 24.0),
    (24, 17.0),
    (18, 12.0),
    (16, 10.0),
    (8, 4.5),
)


def available_memory_gb() -> Optional[float]:
    """Memory macOS can hand out right now (psutil's virtual_memory().available), in GiB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def budget_8gb(available_gb: Optional[float]) -> float:
    """8 GB Macs: sized to the free memory at launch, between a 4.0 GB floor and a 5.0 GB ceiling.
      more than 5.5 GB free -> 5.0 GB
      4.5 to 5.5 GB free    -> 4.5 GB
      less than 4.5 GB free -> 4.0 GB (the floor: below it the engine can't run)
    Unknown free memory -> 4.5 GB (the old fixed value)."""
    if available_gb is None:
        return 4.5
    if available_gb > 5.5:
        return 5.0
    if available_gb >= 4.5:
        return 4.5
    return 4.0


def default_budget_gb(ram_gb: float, available_gb: Optional[float] = None) -> float:
    for size, target in BUDGET_TIERS:
        if ram_gb >= size - 0.5:        # sysconf can report a little under the label
            if size == 8:               # the 8 GB tier follows the free memory at launch
                return budget_8gb(available_gb)
            return target
    return budget_8gb(available_gb)


# config.json in the data folder (~/Library/Application Support/YanahChat/ in the app):
# what the launcher detected and the memory target it gave the engine. Rewritten every launch;
# to pin a different target, set "budget_gb" in settings.json.
def memory_config_path() -> Path:
    return data_dir() / "config.json"


def write_memory_config(ram_gb: float, process_target_gb: float, manual: bool = False,
                        available_gb: Optional[float] = None) -> None:
    cfg = {"ram_gb": int(round(ram_gb)), "process_target_gb": float(process_target_gb),
           "source": "settings.json budget_gb" if manual else
                     ("free memory at launch" if ram_gb < 15.5 else "RAM tier")}
    if available_gb is not None:
        cfg["available_gb"] = round(available_gb, 2)
    try:
        memory_config_path().write_text(json.dumps(cfg, indent=2))
    except Exception as e:                      # never stop the app over this file
        print(f"[yanah] could not write {memory_config_path()}: {e}", file=sys.stderr)


def read_memory_config() -> dict:
    try:
        return json.loads(memory_config_path().read_text())
    except Exception:
        return {}


def default_max_context(ram_gb: float) -> int:
    return 8192 if ram_gb <= 8.5 else 16384


def default_cache_limit_gb(ram_gb: float) -> float:
    return 0.25 if ram_gb <= 8.5 else 0.5


# ------------------------------------------------------------------ models
def dense_ok(d: Path) -> bool:
    return ((d / "config.json").is_file() and (d / "tokenizer_config.json").is_file()
            and any(d.glob("model*.safetensors")))


def expert_preference(settings: dict, ram_gb: float) -> Tuple[str, ...]:
    choice = settings.get("experts", "auto")
    if choice in EXPERT_DIRS:
        return (choice,)
    # 2-bit records are 0.67x the size: 1.5x the experts in the same pool,
    # which matters most on small Macs. With RAM to spare, prefer 4-bit quality.
    return ("4bit", "2bit") if ram_gb > 16.5 else ("2bit", "4bit")


def _log_check(msg: str) -> None:
    """Model checks go to the console / the app's log (logs/app.log in Yanah.app)."""
    print(f"[models] {msg}", file=sys.stderr, flush=True)


def experts_present(d: Path) -> bool:
    """The startup check: is the weight file, experts.bin, there? (Not just the folder.)
    A size that doesn't match experts.json is logged as a warning (a copy still in
    progress, or a damaged file) but doesn't trigger a new download."""
    b = d / "experts.bin"
    if not b.is_file():
        _log_check(f"checking {b}: not found")
        return False
    size = b.stat().st_size
    _log_check(f"checking {b}: found ({size / 2**30:.2f} GiB)")
    try:
        spec = json.loads((d / "experts.json").read_text())
        want = (spec["n_layers"] + spec["n_mtp_layers"]) * spec["n_experts"] * spec["slot_bytes"]
        if size != want:
            _log_check(f"WARNING {b} is {size:,} bytes; experts.json expects {want:,} "
                       "(still copying, or damaged?). Using it anyway.")
    except FileNotFoundError:
        _log_check(f"WARNING {d / 'experts.json'} is missing; the engine needs it next to experts.bin")
    except Exception as e:
        _log_check(f"WARNING couldn't read {d / 'experts.json'}: {e}")
    return True


# The installed model, at fixed paths (what the downloader writes)
DENSE_PATH = APP_SUPPORT / "models" / "Qwen3-80B-dense"
EXPERTS_2BIT_PATH = APP_SUPPORT / "models" / "Qwen3-80B-2bit-experts"


def check_models(settings: Optional[dict] = None) -> Optional[Dict[str, object]]:
    """Startup check: is the model installed? Logs (to logs/app.log in the app) every
    path it checks and what it found.

    1. Exactly ~/Library/Application Support/YanahChat/models/Qwen3-80B-dense/model.safetensors
       and .../Qwen3-80B-2bit-experts/experts.bin. Both there: use them, no download,
       whatever settings.json says about 2-bit or 4-bit.
    2. Otherwise the wider search (find_models): the 4-bit set, YANAH_MODELS, and from
       source ~/Desktop/yanah_models."""
    dense_file = DENSE_PATH / "model.safetensors"
    experts_file = EXPERTS_2BIT_PATH / "experts.bin"
    _log_check(f"app: {sys.executable} (frozen={FROZEN}), data folder {APP_SUPPORT}")
    d_ok, e_ok = dense_file.is_file(), experts_file.is_file()
    _log_check(f"checking {dense_file}: {'found' if d_ok else 'NOT FOUND'}")
    _log_check(f"checking {experts_file}: {'found' if e_ok else 'NOT FOUND'}")
    if d_ok and e_ok:
        for extra in (DENSE_PATH / "config.json", DENSE_PATH / "tokenizer_config.json",
                      EXPERTS_2BIT_PATH / "experts.json"):
            if not extra.is_file():
                _log_check(f"WARNING {extra} is missing; the engine needs it")
        experts_present(EXPERTS_2BIT_PATH)          # logs the size against experts.json
        _log_check(f"model found: launching the engine with {DENSE_PATH} + {EXPERTS_2BIT_PATH}")
        return {"dense": DENSE_PATH, "experts": EXPERTS_2BIT_PATH, "bits": "2bit"}
    for p in (DENSE_PATH, EXPERTS_2BIT_PATH, APP_SUPPORT / "models"):
        try:
            _log_check(f"contents of {p}: {sorted(x.name for x in p.iterdir()) if p.is_dir() else 'no such folder'}")
        except OSError as e:
            _log_check(f"can't list {p}: {e}")
    _log_check("trying the other model locations")
    return find_models(settings)


def find_models(settings: Optional[dict] = None) -> Optional[Dict[str, object]]:
    """{'dense': Path, 'experts': Path, 'bits': '2bit'|'4bit'} or None.
    Every path it checks is logged, with what it found."""
    settings = settings or load_settings()
    roots = [r for r in model_roots() if r.is_dir()]
    _log_check(f"model folders: {[str(r) for r in model_roots()]} (existing: {[str(r) for r in roots]})")
    dense = None
    for r in roots:
        for n in DENSE_DIRS:
            ok = dense_ok(r / n)
            _log_check(f"checking {r / n}: {'found' if ok else 'not found (needs config.json, tokenizer_config.json, model*.safetensors)'}")
            if ok:
                dense = r / n
                break
        if dense:
            break
    if dense is None:
        _log_check("no dense weights found: download needed")
        return None
    _log_check(f"experts setting: {settings.get('experts', 'auto')!r}, RAM {total_ram_gb():.0f} GB")
    for bits in expert_preference(settings, total_ram_gb()):
        for r in roots:
            e = r / EXPERT_DIRS[bits]
            if experts_present(e):
                _log_check(f"using {dense} + {e}")
                return {"dense": dense, "experts": e, "bits": bits}
    _log_check("no experts.bin found: download needed")
    return None


def wanted_download(settings: Optional[dict] = None) -> str:
    """Which expert set a first-run download fetches."""
    settings = settings or load_settings()
    return expert_preference(settings, total_ram_gb())[0] if settings.get("experts") in EXPERT_DIRS else "2bit"


# ------------------------------------------------------------------- ports
def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if nothing is listening and we could bind it. No SO_REUSEADDR,
    so this is stricter than the Flask server's own bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(start: int, avoid=(), tries: int = 200, host: str = "127.0.0.1") -> int:
    for p in range(start, min(start + tries, 65535)):
        if p not in avoid and port_is_free(p, host):
            return p
    # last resort: let the OS pick one
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
