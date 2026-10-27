#!/bin/bash
# 2-bit g32 expert requantization, run natively (Metal). Resumable: if this
# window is closed mid-way, double-click again to continue from the last layer.
cd "/Users/admin/Desktop/yanah_build"
source .venv/bin/activate
echo "==> quantizing experts to 2-bit g32 -> ~/Desktop/yanah_models/Qwen3-80B-2bit-experts"
echo "    log: ~/quantize_2bit.log"
python quantize_2bit.py 2>&1 | tee ~/quantize_2bit.log
echo
echo "Finished (exit ${PIPESTATUS[0]}). You can close this window."
