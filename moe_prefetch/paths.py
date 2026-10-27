# SPDX-License-Identifier: FSL-1.1-ALv2
"""
Where the models live. One place, so every script agrees.

    ~/Desktop/yanah_models/
        Qwen3-80B-4bit/            MLX 4-bit checkpoint (dense weights, tokenizer, config)
        Qwen3-80B-4bit-experts/    experts.bin + experts.json from repack.py
        Qwen3-80B-2bit-experts/    experts.bin + experts.json from quantize_2bit.py

Set YANAH_MODELS to move the whole tree somewhere else.
"""
import os
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("YANAH_MODELS", "~/Desktop/yanah_models")).expanduser()
MODEL_4BIT = MODELS_ROOT / "Qwen3-80B-4bit"
EXPERTS_4BIT = MODELS_ROOT / "Qwen3-80B-4bit-experts"
EXPERTS_2BIT = MODELS_ROOT / "Qwen3-80B-2bit-experts"
