# PyInstaller spec for Yanah.app.  Build with:  ./build_app.command
#
# One-folder app (not --onefile): a --onefile .app unpacks ~400 MB to a temp
# folder on every launch, and PyInstaller has deprecated onefile + windowed
# on macOS. A one-folder bundle starts instantly and is signed as a normal app.
#
# Everything the app needs is inside it: the Python interpreter, mlx (+ its
# Metal kernels, mlx.metallib), mlx_lm (its model classes are imported by
# name at runtime, so they are listed explicitly), transformers/tokenizers,
# flask, httpx, huggingface_hub (+ hf_xet), tkinter, and Yanah's own code.
# Nothing is read from the user's Python, so a missing or broken Python on
# their Mac can't cause ModuleNotFoundError.
import json
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

HERE = os.path.abspath(SPECPATH)
VERSION = os.environ.get("YANAH_VERSION", "0.1.0")

datas, binaries, hiddenimports = [], [], []

# mlx: libmlx.dylib + mlx.metallib live in mlx/lib and must stay together
# (libmlx finds its kernels next to itself). collect_all keeps the layout.
for pkg in ("mlx", "mlx_lm", "hf_xet", "tokenizers", "webview"):   # webview: the chat window (pywebview)
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# mlx's C++ headers and cmake files are only for building extensions
_skip = ("/mlx/include/", "/mlx/lib/cmake/", "/mlx/share/")
datas = [(src, dst) for src, dst in datas if not any(k in src.replace(os.sep, "/") for k in _skip)]

hiddenimports += collect_submodules("mlx_lm.models")          # loaded via importlib by model_type
hiddenimports += collect_submodules("moe_prefetch")
hiddenimports += [
    "mlx_server", "yanah_shell", "download_models", "yanah_runtime",
    "flask_cors", "httpx", "jinja2", "safetensors",
    "tkinter", "tkinter.ttk", "_tkinter",
    "huggingface_hub", "huggingface_hub.hf_api",
    "psutil",                     # free memory at launch (8 GB Macs size the engine by it)
    "webview", "webview.platforms.cocoa",   # the chat window: macOS WebKit, no browser needed
]
# Qwen3-Next's tokenizer is transformers' Qwen2 tokenizer, found by name at runtime
hiddenimports += collect_submodules("transformers.models.qwen2")

# transformers checks installed versions of its dependencies at import time
# (importlib.metadata), so their metadata must ship too.
for pkg in ("transformers", "tokenizers", "huggingface_hub", "safetensors", "tqdm", "numpy",
            "packaging", "filelock", "regex", "pyyaml", "jinja2", "mlx", "mlx_lm", "flask"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Yanah's files (read-only defaults; editable copies go to Application Support)
for f in ("index.html", "welcome.md", "user_profile.md", "icon.png", "yanah_logo.png"):
    datas.append((os.path.join(HERE, f), "."))
# persona folders (personas/default, personas/stoic, personas/muse, ...), kept as folders
datas.append((os.path.join(HERE, "personas"), "personas"))
# this project's licence -> Contents/Resources/LICENSE
datas.append((os.path.join(HERE, "LICENSE"), "."))

# Which Hugging Face repo the first-run download uses (set by build_app.command)
cfg = os.path.join(HERE, "build", "yanah_build_config.json")
os.makedirs(os.path.dirname(cfg), exist_ok=True)
with open(cfg, "w") as fh:
    json.dump({"model_repo": os.environ.get("MODEL_REPO", ""), "version": VERSION}, fh)
datas.append((cfg, "."))

a = Analysis(
    [os.path.join(HERE, "start_yanah.py")],
    pathex=[HERE],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "jax", "flax", "matplotlib", "IPython", "notebook",
              "pandas", "scipy", "sklearn", "PIL.ImageQt", "PyQt5", "PyQt6", "PySide6",
              "pytest", "moe_prefetch.test_synthetic", "moe_prefetch.test_mlx_backend",
              "moe_prefetch.test_cache_sim"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Yanah",
    console=False,                  # a windowed app: no Terminal
    argv_emulation=False,
    target_arch="arm64",            # mlx is Apple Silicon only
    codesign_identity=os.environ.get("SIGN_IDENTITY") or None,
    entitlements_file=os.path.join(HERE, "entitlements.plist") if os.environ.get("SIGN_IDENTITY") else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Yanah", strip=False, upx=False)
app = BUNDLE(
    coll,
    name="Yanah.app",
    icon=os.path.join(HERE, "yanah_logo.icns"),    # the YN monogram, 16–1024 px
    bundle_identifier="app.yanah.desktop",
    version=VERSION,
    info_plist={
        "CFBundleName": "Yanah",
        "CFBundleDisplayName": "Yanah",
        "CFBundleIconFile": "yanah_logo.icns",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "14.0",
        "LSArchitecturePriority": ["arm64"],
        "LSUIElement": False,
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHumanReadableCopyright": "Model: Qwen3-Next-80B-A3B-Instruct, Apache-2.0",
    },
)
