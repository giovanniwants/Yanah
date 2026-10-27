"""
Package Yanah's models for download, and (optionally) upload them to your
Hugging Face account. Yanah.app downloads from that repository on first run.

    python publish_models.py                       # build ~/Desktop/yanah_publish (no upload)
    hf auth login                                  # once, with a WRITE token
    python publish_models.py --repo YOURNAME/Yanah-Qwen3-Next-80B --upload

What it makes (see download_models.py for why it's shaped this way):

    dense/model.safetensors        only the non-expert tensors of the 4-bit
                                   checkpoint (1.3 GiB of its 42 GiB); the
                                   engine streams experts from experts.bin and
                                   never reads the stacked expert tensors
    dense/config.json, tokenizer files, chat template
    experts-2bit/experts.json + parts/layer-NNN.bin (one per layer)
    experts-4bit/...               only with --with-4bit (another 41 GiB)
    yanah_manifest.json            sizes + sha256 of everything
    README.md                      model card (Apache-2.0, from Qwen3-Next-80B-A3B-Instruct)
    LICENSE                        the base model's license, from the Qwen repository

The upload creates the repo (public) if it doesn't exist and sends the folder
with HfApi.upload_folder, which resumes when re-run. The repo id is printed
at the end, for model_repo.txt.

Needs ~29 GiB free for the 2-bit set (the parts are copies).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

DENSE_EXTRA = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
               "chat_template.jinja")


def is_expert(name: str) -> bool:
    return ".switch_mlp." in name or ".mlp.experts." in name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slim_safetensors(src_dir: Path, out_file: Path) -> dict:
    """Copy every non-expert tensor, byte for byte, into one safetensors file."""
    tensors = []                                   # (name, file, entry, data_start)
    meta = None
    for f in sorted(src_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        meta = meta or hdr.get("__metadata__")
        for k, v in hdr.items():
            if k != "__metadata__" and not is_expert(k):
                tensors.append((k, f, v, 8 + n))
    if not tensors:
        raise SystemExit(f"no safetensors in {src_dir}")
    tensors.sort(key=lambda t: t[0])
    header, off = {}, 0
    for k, _, v, _ in tensors:
        size = v["data_offsets"][1] - v["data_offsets"][0]
        header[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + size]}
        off += size
    if meta:
        header["__metadata__"] = meta
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)                    # 8-byte aligned data start, as safetensors writers do
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for k, f, v, base in tensors:
            b, e = v["data_offsets"]
            with open(f, "rb") as src:
                src.seek(base + b)
                left = e - b
                while left:
                    chunk = src.read(min(left, 16 << 20))
                    if not chunk:
                        raise IOError(f"{f} is truncated at {k}")
                    out.write(chunk)
                    left -= len(chunk)
    return {"tensors": len(tensors), "bytes": off}


def split_experts(exp_dir: Path, out_dir: Path) -> dict:
    spec = json.loads((exp_dir / "experts.json").read_text())
    L = spec["n_layers"] + spec["n_mtp_layers"]
    layer_bytes = spec["n_experts"] * spec["slot_bytes"]
    src = exp_dir / "experts.bin"
    if src.stat().st_size != L * layer_bytes:
        raise SystemExit(f"{src} is {src.stat().st_size} bytes, expected {L * layer_bytes}")
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    keep = {k: spec[k] for k in spec if k not in ("source", "rel_rms_error_vs_source")}
    if "scheme" in spec:
        keep["scheme"] = spec["scheme"]
    (out_dir / "experts.json").write_text(json.dumps(keep, indent=1))
    parts = []
    with open(src, "rb") as f:
        for i in range(L):
            p = out_dir / "parts" / f"layer-{i:03d}.bin"
            if not (p.exists() and p.stat().st_size == layer_bytes):
                f.seek(i * layer_bytes)
                with open(p, "wb") as out:
                    left = layer_bytes
                    while left:
                        chunk = f.read(min(left, 16 << 20))
                        out.write(chunk)
                        left -= len(chunk)
            parts.append(p)
            sys.stderr.write(f"\r  {out_dir.name}: layer {i + 1}/{L}")
    sys.stderr.write("\n")
    return {"layer_bytes": layer_bytes, "parts": parts}


def entry(root: Path, p: Path) -> dict:
    return {"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}


CARD = """---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen3-Next-80B-A3B-Instruct
library_name: mlx
tags: [mlx, moe, yanah]
---
# Yanah model files (Qwen3-Next-80B-A3B-Instruct, streamed-experts layout)

Files for the Yanah Mac app, which streams the mixture-of-experts weights
from SSD instead of holding them in RAM.

* `dense/`: the non-expert weights of mlx-community's 4-bit MLX conversion
  (attention, Gated DeltaNet, shared experts, embeddings, routers), the
  tokenizer and config.
* `experts-2bit/`: routed experts requantized from 4-bit g64 to 2-bit g32
  (affine), one record per expert, split into one file per layer.
* `yanah_manifest.json`: size and sha256 of every file.

Derived from Qwen/Qwen3-Next-80B-A3B-Instruct (Apache-2.0). Quantization
reduces quality versus the original model.
"""


QWEN_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# --coder: the same layout for YanahCoder.app (Qwen3-Coder-Next, 4-bit experts only)
CODER_REPO = "Qwen/Qwen3-Coder-Next"
CODER_CARD = """---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/LICENSE
base_model: Qwen/Qwen3-Coder-Next
library_name: mlx
tags: [mlx, moe, yanah, coder]
---
# YanahCoder model files (Qwen3-Coder-Next, streamed-experts layout)

Files for the YanahCoder Mac app, which streams the mixture-of-experts
weights from SSD instead of holding them in RAM.

* `dense/`: the non-expert weights of a 4-bit MLX conversion (attention,
  Gated DeltaNet, shared experts, embeddings, routers), the tokenizer and config.
* `experts-4bit/`: routed experts (4-bit), one record per expert, split into
  one file per layer.
* `yanah_manifest.json`: size and sha256 of every file.

Derived from Qwen/Qwen3-Coder-Next (Apache-2.0). Quantization reduces quality
versus the original model.
"""


def fetch_license(out: Path, local: str = None) -> Path:
    """Put the base model's LICENSE (Apache-2.0) in the upload folder: from
    --license if given, else downloaded from the Qwen repository."""
    dst = out / "LICENSE"
    if local:
        shutil.copyfile(Path(local).expanduser(), dst)
    else:
        from huggingface_hub import hf_hub_download
        try:
            src = hf_hub_download(QWEN_REPO, "LICENSE")
        except Exception as e:
            raise SystemExit(f"could not download LICENSE from {QWEN_REPO} ({type(e).__name__}: {e}). "
                             "Download it yourself and pass --license PATH.")
        shutil.copyfile(src, dst)
    head = dst.read_text(errors="replace")[:400]
    if "Apache License" not in head:
        raise SystemExit(f"{dst} doesn't look like the Apache-2.0 license; check it before publishing")
    print(f"  LICENSE from {local or QWEN_REPO}", file=sys.stderr)
    return dst


def upload(folder: Path, repo_id: str) -> str:
    """Create the repo if needed (public, so the app can download without a
    token) and upload the folder. upload_folder is resumable: if it stops,
    run the same command again and finished files are skipped."""
    from huggingface_hub import HfApi
    api = HfApi()
    url = api.create_repo(repo_id, private=False, repo_type="model", exist_ok=True)
    repo_id = url.repo_id                    # canonical name as the Hub reports it
    print(f"[4] uploading {folder} to https://huggingface.co/{repo_id}", file=sys.stderr)
    info = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                             commit_message="Yanah model files",
                             ignore_patterns=[".cache/**", "**/.DS_Store"])
    print(f"  commit: {getattr(info, 'commit_url', info)}", file=sys.stderr)
    return repo_id


def main():
    home = Path("~").expanduser()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", default=str(home / "Desktop/yanah_models/Qwen3-80B-4bit"))
    ap.add_argument("--experts-2bit", default=str(home / "Desktop/yanah_models/Qwen3-80B-2bit-experts"))
    ap.add_argument("--experts-4bit", default=str(home / "Desktop/yanah_models/Qwen3-80B-4bit-experts"))
    ap.add_argument("--with-4bit", action="store_true")
    ap.add_argument("--out", default=str(home / "Desktop/yanah_publish"))
    ap.add_argument("--repo", default=None, help="YOURNAME/REPO on Hugging Face")
    ap.add_argument("--upload", action="store_true", help="upload --out to --repo (needs `hf auth login`)")
    ap.add_argument("--license", default=None,
                    help="path to the Qwen LICENSE file (default: download it from Qwen/Qwen3-Next-80B-A3B-Instruct)")
    ap.add_argument("--coder", action="store_true",
                    help="YanahCoder: Qwen3-Coder-Next from ~/Desktop/yanah_coder_models, 4-bit experts only")
    a = ap.parse_args()
    global QWEN_REPO, CARD
    if a.coder:                        # different defaults; anything given explicitly still wins
        cm = home / "Desktop/yanah_coder_models"
        if a.dense == ap.get_default("dense"):
            a.dense = str(cm / "Qwen3-Coder-Next-4bit")
        if a.experts_4bit == ap.get_default("experts_4bit"):
            a.experts_4bit = str(cm / "Qwen3-Coder-Next-4bit-experts")
        if a.out == ap.get_default("out"):
            a.out = str(home / "Desktop/yanah_coder_publish")
        QWEN_REPO, CARD = CODER_REPO, CODER_CARD
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dense_src = Path(a.dense).expanduser()

    print("[1] dense weights", file=sys.stderr)
    info = slim_safetensors(dense_src, out / "dense" / "model.safetensors")
    print(f"  {info['tensors']} tensors, {info['bytes'] / 2**30:.2f} GiB", file=sys.stderr)
    for n in DENSE_EXTRA:
        if (dense_src / n).exists():
            shutil.copy2(dense_src / n, out / "dense" / n)
    manifest = {"version": 1, "base_model": QWEN_REPO,
                "dense": {"files": [entry(out, p) for p in sorted((out / "dense").iterdir()) if p.is_file()]},
                "experts": {}}
    sets = [] if a.coder else [("2bit", Path(a.experts_2bit).expanduser())]
    if a.with_4bit or a.coder:
        sets.append(("4bit", Path(a.experts_4bit).expanduser()))
    for bits, src in sets:
        print(f"[2] experts {bits}", file=sys.stderr)
        r = split_experts(src, out / f"experts-{bits}")
        print("  hashing parts", file=sys.stderr)
        manifest["experts"][bits] = {"layer_bytes": r["layer_bytes"],
                                     "json": entry(out, out / f"experts-{bits}" / "experts.json"),
                                     "parts": [entry(out, p) for p in r["parts"]]}
    (out / "yanah_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.md").write_text(CARD)
    fetch_license(out, a.license)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[3] {out}: {total / 1e9:.1f} GB ready", file=sys.stderr)

    if a.upload:
        if not a.repo:
            raise SystemExit("--upload needs --repo YOURNAME/REPO")
        if not (out / "LICENSE").is_file():
            raise SystemExit(f"{out / 'LICENSE'} is missing; refusing to publish without the model's license")
        repo_id = upload(out, a.repo)
        print(f"  done. Put this in model_repo.txt and rebuild the app:", file=sys.stderr)
        print(repo_id)
    else:
        print("not uploaded. To publish: hf auth login, then rerun with --repo YOURNAME/REPO --upload",
              file=sys.stderr)


if __name__ == "__main__":
    main()
