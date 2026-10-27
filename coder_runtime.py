"""
Settings for YanahCoder.app: where its files live, how much memory the engine
gets, which model folders count, and where the bundled OpenCode is.

Kept apart from yanah_runtime.py so the chat app (Yanah.app) and the coding
app never share a data folder, a model folder or settings. The generic
helpers (RAM size, free ports, model-folder checks) come from yanah_runtime.

Where things live
-----------------
data      ~/Library/Application Support/YanahCoder   settings.json, config.json, logs,
                                                     run_opencode.command, OpenCode's own state
models    $YANAH_CODER_MODELS, else ~/Desktop/yanah_coder_models (made by
          prepare_coder_models.command), else ~/Library/Application Support/YanahCoder/models
          (first-run downloads go there)
opencode  YanahCoder.app/Contents/Resources/OpenCode.app, the OpenCode desktop app
          (from source: build/opencode-desktop/OpenCode.app, put there by build_coder.command)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import yanah_runtime as yrt
from yanah_runtime import FROZEN, dense_ok, experts_ok, find_free_port, total_ram_gb  # noqa: F401

APP_NAME = "YanahCoder"
APP_SUPPORT = Path("~/Library/Application Support").expanduser() / APP_NAME
DESKTOP_MODELS = Path("~/Desktop/yanah_coder_models").expanduser()

# prepare_coder_models.command writes the first names; a first-run download
# (download_models.py, unchanged) writes the second ones into the models folder.
DENSE_DIRS = ("Qwen3-Coder-Next-4bit", "Qwen3-80B-dense")
EXPERT_DIRS = ("Qwen3-Coder-Next-4bit-experts", "Qwen3-80B-4bit-experts")

MODEL_ID = "qwen3-coder-next"
MIN_RAM_GB = 16

# --budget-gb (the engine's process memory target) by the Mac's physical RAM.
# Sizes between tiers use the tier below (e.g. 36 GB -> 26). 64 GB and up -> 54.
BUDGET_TIERS = (     # (RAM GB, --budget-gb)
    (64, 54.0),
    (48, 40.0),
    (32, 26.0),
    (24, 18.0),
    (18, 14.0),
    (16, 10.0),
)

DEFAULT_SETTINGS = {
    "model_repo": None,      # Hugging Face repo for the first-run download
    "budget_gb": None,       # None -> BUDGET_TIERS
    "max_context": None,     # None -> by RAM
    "engine_port": 8888,
    "workspace": None,       # the folder OpenCode opens in (None -> ~/YanahCoder)
}


# ------------------------------------------------------------------ places
def resources_dir() -> Path:
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    env = os.environ.get("YANAH_CODER_DATA")
    d = Path(env).expanduser() if env else APP_SUPPORT
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_roots() -> list:
    env = os.environ.get("YANAH_CODER_MODELS")
    if env:
        return [Path(env).expanduser()]
    return [DESKTOP_MODELS, APP_SUPPORT / "models"]


def download_root() -> Path:
    env = os.environ.get("YANAH_CODER_MODELS")
    return Path(env).expanduser() if env else APP_SUPPORT / "models"


def opencode_app_path() -> Path:
    """The bundled OpenCode desktop app."""
    env = os.environ.get("YANAH_OPENCODE_APP")
    if env:
        return Path(env).expanduser()
    if FROZEN:   # .../YanahCoder.app/Contents/MacOS/YanahCoder -> Contents/Resources/OpenCode.app
        return Path(sys.executable).resolve().parents[1] / "Resources" / "OpenCode.app"
    return Path(__file__).resolve().parent / "build" / "opencode-desktop" / "OpenCode.app"


def system_prompt_path() -> Path:
    return resources_dir() / "coder_system.md"


# ---------------------------------------------------------------- settings
def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    p = data_dir() / "settings.json"
    try:
        s.update(json.loads(p.read_text()))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[yanahcoder] ignoring unreadable {p}: {e}", file=sys.stderr)
    return s


def save_setting(key: str, value) -> None:
    p = data_dir() / "settings.json"
    try:
        cur = json.loads(p.read_text())
    except Exception:
        cur = {}
    cur[key] = value
    p.write_text(json.dumps(cur, indent=2))


def model_repo(settings: Optional[dict] = None) -> str:
    if os.environ.get("YANAH_CODER_MODEL_REPO"):
        return os.environ["YANAH_CODER_MODEL_REPO"]
    if settings and settings.get("model_repo"):
        return settings["model_repo"]
    try:
        cfg = json.loads((resources_dir() / "coder_build_config.json").read_text())
        return cfg.get("model_repo") or ""
    except Exception:
        return ""


# ------------------------------------------------------------------ memory
def default_budget_gb(ram_gb: float) -> Optional[float]:
    """None below 16 GB: the 4-bit coder model isn't supported there."""
    for size, budget in BUDGET_TIERS:
        if ram_gb >= size - 0.5:
            return budget
    return None


def default_max_context(ram_gb: float) -> int:
    """OpenCode's system prompt and tool list alone are ~4-6k tokens, and code is long.
    KV memory is small on this model (12 attention layers: ~25 KB per token)."""
    return 32768 if ram_gb < 23.5 else 65536


def write_memory_config(ram_gb: float, budget_gb: float, manual: bool) -> None:
    cfg = {"ram_gb": int(round(ram_gb)), "process_target_gb": float(budget_gb),
           "source": "settings.json budget_gb" if manual else "RAM tier"}
    try:
        (data_dir() / "config.json").write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"[yanahcoder] could not write config.json: {e}", file=sys.stderr)


# ------------------------------------------------------------------ models
def find_models() -> Optional[Dict[str, Path]]:
    """{'dense': Path, 'experts': Path} or None. 4-bit experts only."""
    roots = [r for r in model_roots() if r.is_dir()]
    dense = next((r / n for r in roots for n in DENSE_DIRS if dense_ok(r / n)), None)
    experts = next((r / n for r in roots for n in EXPERT_DIRS if experts_ok(r / n)), None)
    if dense is None or experts is None:
        return None
    return {"dense": dense, "experts": experts}


def is_coder_model(dense: Path) -> bool:
    """True if the checkpoint's chat template uses Qwen3-Coder's tool-call format
    (<function=...>). Catches pointing YanahCoder at Yanah's chat model by mistake."""
    text = ""
    for n in ("chat_template.jinja", "tokenizer_config.json"):
        try:
            text += (dense / n).read_text(encoding="utf-8")
        except OSError:
            pass
    return "<function=" in text or "<parameter=" in text


__all__ = [n for n in dir() if not n.startswith("_")] + ["yrt"]
