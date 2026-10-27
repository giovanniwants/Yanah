# PyInstaller spec for YanahCoder.app.  Build with:  ./build_coder.command
#
# Same engine as Yanah.app (mlx_server + moe_prefetch, everything bundled, nothing
# read from the user's Python), with the coder launcher (start_coder.py) instead
# of the chat. The OpenCode desktop app is not part of this spec: build_coder.command
# downloads the pinned version, checks its hash and copies OpenCode.app to
# YanahCoder.app/Contents/Resources/OpenCode.app after PyInstaller has run.
import json
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

HERE = os.path.abspath(SPECPATH)
VERSION = os.environ.get("YANAH_CODER_VERSION", "0.1.0")

datas, binaries, hiddenimports = [], [], []

# mlx: libmlx.dylib + mlx.metallib must stay together; collect_all keeps the layout.
for pkg in ("mlx", "mlx_lm", "hf_xet", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
_skip = ("/mlx/include/", "/mlx/lib/cmake/", "/mlx/share/")
datas = [(src, dst) for src, dst in datas if not any(k in src.replace(os.sep, "/") for k in _skip)]

hiddenimports += collect_submodules("mlx_lm.models")          # loaded via importlib by model_type
hiddenimports += collect_submodules("moe_prefetch")
hiddenimports += [
    "mlx_server", "tool_calls", "coder_runtime", "yanah_runtime", "start_yanah", "download_models",
    "flask", "httpx", "jinja2", "safetensors",
    "tkinter", "tkinter.ttk", "tkinter.filedialog", "_tkinter",
    "huggingface_hub", "huggingface_hub.hf_api",
]
hiddenimports += collect_submodules("transformers.models.qwen2")   # Qwen3's tokenizer class

for pkg in ("transformers", "tokenizers", "huggingface_hub", "safetensors", "tqdm", "numpy",
            "packaging", "filelock", "regex", "pyyaml", "jinja2", "mlx", "mlx_lm", "flask"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# YanahCoder's files
for f in ("coder_system.md", "yanah_logo.png"):
    datas.append((os.path.join(HERE, f), "."))
# this project's licence -> Contents/Resources/LICENSE
datas.append((os.path.join(HERE, "LICENSE"), "."))

# Which Hugging Face repo the first-run download uses (set by build_coder.command)
cfg = os.path.join(HERE, "build", "coder_build_config.json")
os.makedirs(os.path.dirname(cfg), exist_ok=True)
with open(cfg, "w") as fh:
    json.dump({"model_repo": os.environ.get("CODER_MODEL_REPO", ""), "version": VERSION,
               "opencode": os.environ.get("OPENCODE_VERSION", "")}, fh)
datas.append((cfg, "."))

a = Analysis(
    [os.path.join(HERE, "start_coder.py")],
    pathex=[HERE],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "jax", "flax", "matplotlib", "IPython", "notebook",
              "pandas", "scipy", "sklearn", "PIL.ImageQt", "PyQt5", "PyQt6", "PySide6",
              "pytest", "yanah_shell", "moe_prefetch.test_synthetic", "moe_prefetch.test_mlx_backend",
              "moe_prefetch.test_cache_sim"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="YanahCoder",
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=os.environ.get("SIGN_IDENTITY") or None,
    entitlements_file=os.path.join(HERE, "entitlements.plist") if os.environ.get("SIGN_IDENTITY") else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="YanahCoder", strip=False, upx=False)
app = BUNDLE(
    coll,
    name="YanahCoder.app",
    icon=os.path.join(HERE, "yanah_logo.icns"),
    bundle_identifier="app.yanah.coder",
    version=VERSION,
    info_plist={
        "CFBundleName": "YanahCoder",
        "CFBundleDisplayName": "Yanah Coder",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "14.0",
        "LSArchitecturePriority": ["arm64"],
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "NSHumanReadableCopyright": "Model: Qwen3-Coder-Next, Apache-2.0. Agent: OpenCode, MIT.",
    },
)
