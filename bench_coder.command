#!/bin/bash
# Speed test for YanahCoder's engine against stock MLX (see bench_coder.py).
# Quit YanahCoder first: both need most of the Mac's memory.
cd "$(dirname "$0")" || exit 1
if pgrep -f "Contents/MacOS/YanahCoder" >/dev/null || lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "YanahCoder (or something on port 8888) is running. Quit it, then run this again."
  read -n1 -p "Press any key to close."; exit 1
fi
source .venv/bin/activate || { echo "no .venv here"; read -n1; exit 1; }
python bench_coder.py
echo; cat build/BENCH_RESULT.txt
read -n1 -t 3600 -p "Press any key to close."
