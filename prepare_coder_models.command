#!/bin/bash
# Make YanahCoder's model: Qwen3-Coder-Next as an MLX 4-bit checkpoint plus its experts
# repacked for streaming (experts.bin), in ~/Desktop/yanah_coder_models:
#
#   Qwen3-Coder-Next-4bit/            MLX 4-bit checkpoint (dense weights, tokenizer, config)
#   Qwen3-Coder-Next-4bit-experts/    experts.bin + experts.json (moe_prefetch.repack)
#
# Default source: the official Qwen/Qwen3-Coder-Next weights, converted here with mlx_lm
# (4-bit, group size 64: the same format as Yanah's 4-bit model). That downloads the full
# bf16 model (~160 GB) into the Hugging Face cache first; run it on your big Mac.
# To start from an MLX 4-bit conversion you trust instead, set MLX_4BIT_REPO=owner/name.
#
# Needs: this folder's .venv (mlx_lm), and free disk: ~250 GB converting, ~95 GB after.
# Safe to run again: finished steps are skipped.
cd "$(dirname "$0")" || exit 1
set -o pipefail
OUT="${YANAH_CODER_MODELS:-$HOME/Desktop/yanah_coder_models}"
DENSE="$OUT/Qwen3-Coder-Next-4bit"
EXP="$OUT/Qwen3-Coder-Next-4bit-experts"
SRC="${CODER_SOURCE:-/Users/admin/Desktop/yanah_models/Qwen3-Coder-Next}"
LOG=build/prepare_coder_models.log
mkdir -p build "$OUT"
exec > >(tee "$LOG") 2>&1
die() { echo; echo "STOPPED: $*"; echo "Log: $(pwd)/$LOG"; read -n1 -p "Press any key to close."; exit 1; }

source .venv/bin/activate || die "no .venv here"
python -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)" || die "mlx_lm missing from .venv"
FREE=$(df -g "$OUT" | awk 'NR==2{print $4}')
echo "free disk at $OUT: ${FREE} GB"

echo; echo "==> 1/3  MLX 4-bit checkpoint -> $DENSE"
if python -c "import sys, coder_runtime as c; from pathlib import Path; sys.exit(0 if c.dense_ok(Path('$DENSE')) else 1)"; then
  echo "    already there, skipping"
elif [ -n "${MLX_4BIT_REPO:-}" ]; then
  [ "${FREE:-0}" -ge 100 ] || die "need ~100 GB free (have ${FREE} GB)"
  hf download "$MLX_4BIT_REPO" --local-dir "$DENSE" || die "download of $MLX_4BIT_REPO failed (rerun to resume)"
else
  # a local folder is converted in place (~45 GB out + ~41 GB experts); a repo name downloads ~160 GB first
  if [ -d "$SRC" ]; then NEED=100; else NEED=250; fi
  [ "${FREE:-0}" -ge "$NEED" ] || die "converting needs ~$NEED GB free (have ${FREE} GB). Free space, or set MLX_4BIT_REPO."
  rm -rf "$DENSE"     # mlx_lm.convert refuses an existing folder; a half-finished one is useless
  python -m mlx_lm convert --hf-path "$SRC" --mlx-path "$DENSE" -q --q-bits 4 --q-group-size 64 \
    || die "mlx_lm convert failed"
fi
python - "$DENSE" <<'PY' || die "the checkpoint isn't a Qwen3-Next-architecture 4-bit model"
import json, sys
from pathlib import Path
import coder_runtime as c
d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
q = cfg.get("quantization") or cfg.get("quantization_config") or {}
print(f"    model_type={cfg.get('model_type')} layers={cfg.get('num_hidden_layers')} experts={cfg.get('num_experts')} "
      f"top_k={cfg.get('num_experts_per_tok')} quant={q.get('bits')}-bit g{q.get('group_size')}")
assert cfg.get("model_type") == "qwen3_next" and cfg.get("num_experts") == 512 and q.get("bits") == 4
print("    Qwen3-Coder tool-call template:", c.is_coder_model(d))
PY

echo; echo "==> 2/3  repack experts -> $EXP"
if python -c "import sys, coder_runtime as c; from pathlib import Path; sys.exit(0 if c.experts_ok(Path('$EXP')) else 1)"; then
  echo "    already there, skipping"
else
  rm -rf "$EXP"
  python -m moe_prefetch.repack "$DENSE" "$EXP" || die "repack failed"
fi

echo; echo "==> 3/3  check"
python - <<PY || die "check failed"
import os
os.environ["YANAH_CODER_MODELS"] = "$OUT"
import coder_runtime as c
m = c.find_models()
assert m, "YanahCoder doesn't see the model folders"
print(f"    YanahCoder will use: {m['dense']}  +  {m['experts']}")
PY
du -sh "$DENSE" "$EXP"
echo
echo "Done. YanahCoder (and build_coder.command's smoke test) will find the model in $OUT."
echo "The bf16 download (~160 GB) can be deleted to get the space back:"
echo "    rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next"
echo "To let other Macs download it: python publish_models.py --coder --repo YOURNAME/YanahCoder-Qwen3-Coder-Next --upload"
read -n1 -t 600 -p "Press any key to close."
