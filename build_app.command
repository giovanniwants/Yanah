#!/bin/bash
# Build Yanah.app and Yanah-<version>.dmg.  Double-click, or run in Terminal.
#
#   1. checks the toolchain (Apple Silicon, tkinter)
#   2. runs the app tests (tiny model, ~1 min)
#   3. builds dist/Yanah.app with PyInstaller (a few minutes)
#   4. self-test of the BUILT app: imports, Metal GPU, tokenizer
#   5. smoke test of the BUILT app's engine on your real models: loads them,
#      generates a few tokens, reports tok/s
#   6. makes dist/Yanah-<version>.dmg
#
# Download source: put your Hugging Face repo (e.g. yourname/Yanah-Qwen3-Next-80B)
# in model_repo.txt next to this file, or set MODEL_REPO. Without it the app
# still works on Macs that already have ~/Desktop/yanah_models.
# Results: build/BUILD_RESULT.txt, full log: build/build.log
cd "$(dirname "$0")" || exit 1
mkdir -p build
LOG=build/build.log
RESULT=build/BUILD_RESULT.txt
: > "$RESULT"
exec > >(tee "$LOG") 2>&1
step() { echo; echo "==> $*"; }
ok()   { echo "OK   $*" | tee -a "$RESULT"; }
bad()  { echo "FAIL $*" | tee -a "$RESULT"; }
die()  { bad "$*"; echo; echo "Build stopped. Details: $(pwd)/$LOG"; read -n1 -p "Press any key to close."; exit 1; }

source .venv/bin/activate || die "no .venv here"
export MODEL_REPO="${MODEL_REPO:-$(cat model_repo.txt 2>/dev/null | tr -d '[:space:]')}"
export YANAH_VERSION="${YANAH_VERSION:-0.1.0}"

step "toolchain"
[ "$(uname -m)" = "arm64" ] || die "needs an Apple Silicon Mac"
python -c "import platform,sys; assert platform.machine()=='arm64'; print(sys.version)" || die "python is not arm64"
if ! python -c "import tkinter" 2>/dev/null; then
  echo "    tkinter missing from Homebrew Python; installing python-tk@3.14"
  brew install python-tk@3.14 || die "could not install python-tk@3.14 (run: brew install python-tk@3.14)"
  python -c "import tkinter" || die "tkinter still missing after brew install python-tk@3.14"
fi
ok "toolchain: $(python -c 'import sys;print(sys.version.split()[0])'), tkinter $(python -c 'import tkinter;print(tkinter.TkVersion)')"
pip install -q "pyinstaller>=6.16" pillow psutil pywebview || die "pip install failed"
python -c "import PyInstaller; print('pyinstaller', PyInstaller.__version__)" || die "pyinstaller install failed"
[ -n "$MODEL_REPO" ] && ok "MODEL_REPO=$MODEL_REPO" || echo "NOTE no MODEL_REPO: first-run download disabled in this build" | tee -a "$RESULT"

step "app tests"
python test_app.py > build/test_app.log 2>&1 && ok "app tests ($(grep -c '^\[' build/test_app.log) checks)" \
  || die "app tests failed: see build/test_app.log"

step "pyinstaller"
rm -rf dist/Yanah.app dist/Yanah
pyinstaller --noconfirm --clean --log-level WARN Yanah.spec || die "pyinstaller failed"
[ -x dist/Yanah.app/Contents/MacOS/Yanah ] || die "dist/Yanah.app was not created"
ok "built dist/Yanah.app ($(du -sh dist/Yanah.app | cut -f1))"

BIN=dist/Yanah.app/Contents/MacOS/Yanah
step "self-test of the built app"
"$BIN" --role selftest > build/selftest.json 2>build/selftest.err
if [ $? -eq 0 ]; then ok "self-test: $(python -c "import json;d=json.load(open('build/selftest.json'));print(d['gpu'],'|',d['tokenizer'])")"
else cat build/selftest.json build/selftest.err; die "self-test of the built app failed (build/selftest.json)"; fi

step "smoke test: the built app's engine on your models"
PORT=$(python -c "import yanah_runtime as r; print(r.find_free_port(18888))")
M=$(python -c "import yanah_runtime as r; m=r.find_models(); print('' if not m else f\"{m['dense']}|{m['experts']}\")")
if [ -z "$M" ]; then
  echo "NOTE no models in ~/Desktop/yanah_models: smoke test skipped" | tee -a "$RESULT"
else
  DENSE="${M%%|*}"; EXP="${M##*|}"
  echo "    $DENSE + $EXP on port $PORT"
  "$BIN" --role engine --port "$PORT" --model "$DENSE" --experts "$EXP" --budget-gb 5.5 > build/smoke_engine.log 2>&1 &
  EPID=$!
  up=""
  for i in $(seq 1 300); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")
    [ "$code" = "200" ] && { up=1; break; }
    [ "$code" = "500" ] && break
    kill -0 $EPID 2>/dev/null || break
    sleep 1
  done
  if [ -n "$up" ]; then
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24,"temperature":0}' \
      > build/smoke_reply.json
    python - <<'PY' | tee -a "$RESULT"
import json
r = json.load(open("build/smoke_reply.json"))
msg = r["choices"][0]["message"]["content"].strip().replace("\n", " ")
u = r.get("usage", {})
print(f"OK   smoke test: loaded in the built app and replied: {msg[:80]!r} ({u.get('completion_tokens','?')} tokens)")
PY
    curl -s "http://127.0.0.1:$PORT/health" > build/smoke_health.json
  else
    tail -20 build/smoke_engine.log
    kill $EPID 2>/dev/null
    die "the built app's engine did not come up (build/smoke_engine.log)"
  fi
  kill $EPID 2>/dev/null; wait $EPID 2>/dev/null
fi

step "dmg"
./make_dmg.sh || die "dmg failed"
ok "dmg: $(ls dist/*.dmg) ($(du -h dist/*.dmg | cut -f1))"

echo; echo "Done."; cat "$RESULT"
echo "DONE" >> "$RESULT"
read -n1 -t 600 -p "Press any key to close."
