#!/bin/bash
# Build YanahCoder.app and YanahCoder-<version>.dmg.  Double-click, or run in Terminal.
#
#   1. checks the toolchain (Apple Silicon, tkinter, PyInstaller)
#   2. downloads the OpenCode desktop app (pinned version) from OpenCode's GitHub releases,
#      checks its sha512, and extracts OpenCode.app from the .dmg into build/opencode-desktop/
#   3. runs the coder tests (tool-call parser, OpenCode launch script, RAM tiers)
#   4. builds dist/YanahCoder.app with PyInstaller
#   5. puts OpenCode.app at YanahCoder.app/Contents/Resources/OpenCode.app, adapts it
#      (patch_opencode.py: own app ID, no auto-updater), re-signs it, then signs the app
#   6. self-test of the BUILT app: imports, Metal, OpenCode.app present, tool-call parser
#   7. smoke test on your coder model (if ~/Desktop/yanah_coder_models exists): loads it,
#      asks it to use a tool, checks a real tool call comes back
#   8. makes dist/YanahCoder-<version>.dmg
#
# Optional: CODER_MODEL_REPO (or coder_model_repo.txt) = Hugging Face repo made with
# `python publish_models.py --coder --repo ... --upload`; without it, users prepare the
# model with prepare_coder_models.command.  SIGN_IDENTITY / NOTARY_PROFILE as for Yanah.
# Results: build/CODER_RESULT.txt, full log: build/build_coder.log
cd "$(dirname "$0")" || exit 1
mkdir -p build
LOG=build/build_coder.log
RESULT=build/CODER_RESULT.txt
: > "$RESULT"
exec > >(tee "$LOG") 2>&1
step() { echo; echo "==> $*"; }
ok()   { echo "OK   $*" | tee -a "$RESULT"; }
bad()  { echo "FAIL $*" | tee -a "$RESULT"; }
die()  { bad "$*"; echo; echo "Build stopped. Details: $(pwd)/$LOG"; read -n1 -p "Press any key to close."; exit 1; }

export YANAH_CODER_VERSION="${YANAH_CODER_VERSION:-0.1.0}"
export CODER_MODEL_REPO="${CODER_MODEL_REPO:-$(cat coder_model_repo.txt 2>/dev/null | tr -d '[:space:]')}"
# OpenCode desktop app, pinned: the newest release that ships a Mac desktop app, and the sha512
# its release publishes for the file (latest-mac.yml). The 2.x releases have no desktop app yet.
export OPENCODE_VERSION="1.19.3"
OPENCODE_URL="https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-desktop-mac-arm64.dmg"
OPENCODE_SHA512="FTc1j4nLOIHuAXgKBZ/jIPIcan7SgIUHJUXAGE/dbpl2zCdG9uIiF1Ht4GUtAINwSn5+aq5w8wdn7iQcItxL5w=="

APP=dist/YanahCoder.app
BIN=$APP/Contents/MacOS/YanahCoder

source .venv/bin/activate || die "no .venv here (the same one build_app.command uses)"

step "toolchain"
[ "$(uname -m)" = "arm64" ] || die "needs an Apple Silicon Mac"
python -c "import tkinter" 2>/dev/null || die "tkinter missing (brew install python-tk@3.14)"
pip install -q "pyinstaller>=6.16" 2>&1 | grep -v "^\[notice\]" || true
python -c "import PyInstaller, mlx_lm, flask" || die "PyInstaller / mlx_lm / flask missing from .venv"
ok "toolchain: $(python -c 'import sys;print(sys.version.split()[0])'), pyinstaller $(python -c 'import PyInstaller;print(PyInstaller.__version__)')"
[ -n "$CODER_MODEL_REPO" ] && ok "CODER_MODEL_REPO=$CODER_MODEL_REPO" \
  || echo "NOTE no CODER_MODEL_REPO: the app won't offer a download (users run prepare_coder_models.command)" | tee -a "$RESULT"

step "OpenCode desktop app $OPENCODE_VERSION"
OC_DIR=build/opencode-desktop
OC_APP="$OC_DIR/OpenCode.app"
if [ ! -d "$OC_APP" ] || [ "$(cat $OC_DIR/VERSION 2>/dev/null)" != "$OPENCODE_VERSION" ]; then
  rm -rf "$OC_DIR"; mkdir -p "$OC_DIR"
  curl -fL --retry 3 -o "$OC_DIR/opencode-desktop.dmg" "$OPENCODE_URL" || die "could not download $OPENCODE_URL"
  got="$(openssl dgst -sha512 -binary "$OC_DIR/opencode-desktop.dmg" | base64 | tr -d '\n')"
  [ "$got" = "$OPENCODE_SHA512" ] || die "OpenCode download does not match its published sha512 (got $got)"
  MNT="$(pwd)/build/opencode-mount"
  hdiutil detach "$MNT" -quiet 2>/dev/null; rm -rf "$MNT"
  hdiutil attach "$OC_DIR/opencode-desktop.dmg" -nobrowse -readonly -mountpoint "$MNT" -quiet || die "could not open the OpenCode .dmg"
  SRC_APP=$(find "$MNT" -maxdepth 1 -name "*.app" | head -1)
  [ -n "$SRC_APP" ] || { hdiutil detach "$MNT" -quiet; die "no .app inside the OpenCode .dmg"; }
  ditto "$SRC_APP" "$OC_APP"                         # ditto keeps the signature and extended attributes
  hdiutil detach "$MNT" -quiet
  rm -f "$OC_DIR/opencode-desktop.dmg"
  echo "$OPENCODE_VERSION" > "$OC_DIR/VERSION"
  echo "    extracted $(basename "$SRC_APP") -> $OC_APP"
fi
codesign --verify --deep --strict "$OC_APP" || die "OpenCode.app's signature does not verify"
AUTH=$(codesign -dv --verbose=2 "$OC_APP" 2>&1 | grep -m1 "^Authority=" | cut -d= -f2)
OCV=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$OC_APP/Contents/Info.plist")
NOTE=$(spctl --assess --type execute "$OC_APP" 2>&1 && echo "Gatekeeper: accepted" || echo "Gatekeeper: not accepted")
ok "OpenCode.app $OCV (sha512 verified; signed by: ${AUTH:-unsigned}; $NOTE)"
[ -f third_party/OPENCODE_LICENSE.txt ] || die "third_party/OPENCODE_LICENSE.txt missing (OpenCode is MIT: its notice must ship)"

step "coder tests"
python test_coder.py > build/test_coder.log 2>&1 && ok "coder tests ($(grep -c '^ok ' build/test_coder.log) checks)" \
  || { cat build/test_coder.log; die "coder tests failed: see build/test_coder.log"; }

step "pyinstaller"
rm -rf "$APP" dist/YanahCoder
pyinstaller --noconfirm --clean --log-level WARN YanahCoder.spec || die "pyinstaller failed"
[ -x "$BIN" ] || die "$APP was not created"
ok "built $APP ($(du -sh "$APP" | cut -f1))"

step "OpenCode.app into the app, then sign"
if [ -n "${SIGN_IDENTITY:-}" ]; then        # sign everything PyInstaller made BEFORE adding OpenCode,
  codesign --force --deep --options runtime --timestamp --entitlements entitlements.plist \
    --sign "$SIGN_IDENTITY" "$APP" || die "codesign failed"
fi
NESTED="$APP/Contents/Resources/OpenCode.app"
rm -rf "$NESTED"
ditto "$OC_APP" "$NESTED"
cp third_party/OPENCODE_LICENSE.txt "$APP/Contents/Resources/OPENCODE_LICENSE.txt"
# Adapt it for YanahCoder (own app ID so it runs next to a separately installed OpenCode,
# no auto-updater, no opencode:// takeover) and re-sign it: the edit replaces OpenCode's signature.
python patch_opencode.py "$NESTED" || die "could not adapt OpenCode.app (patch_opencode.py)"
if [ -n "${SIGN_IDENTITY:-}" ]; then
  codesign --force --deep --options runtime --timestamp --entitlements entitlements.plist \
    --sign "$SIGN_IDENTITY" "$NESTED" || die "codesign of OpenCode.app failed"
else
  codesign --force --deep --sign - "$NESTED" || die "ad-hoc codesign of OpenCode.app failed"
fi
# ...then re-seal only the outer bundle (no --deep), so OpenCode.app keeps the signature just made.
if [ -n "${SIGN_IDENTITY:-}" ]; then
  codesign --force --options runtime --timestamp --entitlements entitlements.plist --sign "$SIGN_IDENTITY" "$APP" \
    || die "codesign (outer) failed"
else
  codesign --force --sign - "$APP" || die "ad-hoc codesign failed"
fi
codesign --verify --strict "$APP" || die "app signature does not verify"
codesign --verify --deep --strict "$APP/Contents/Resources/OpenCode.app" || die "OpenCode.app's signature broke"
ok "OpenCode.app at Contents/Resources/OpenCode.app, adapted (app.yanah.coder.app, no updater); signatures verify"

step "self-test of the built app"
"$BIN" --role selftest > build/coder_selftest.json 2>build/coder_selftest.err
if [ $? -eq 0 ]; then ok "self-test: $(python -c "import json;d=json.load(open('build/coder_selftest.json'));print(d['gpu'],'|',d['opencode'],'|',d['tool calls'])")"
else cat build/coder_selftest.json build/coder_selftest.err; die "self-test of the built app failed"; fi

step "smoke test: the built app's engine on your coder model, with a tool"
M=$(python -c "import coder_runtime as c; m=c.find_models(); print('' if not m else f\"{m['dense']}|{m['experts']}\")")
if [ -z "$M" ]; then
  echo "NOTE no coder model in ~/Desktop/yanah_coder_models: smoke test skipped (run prepare_coder_models.command)" | tee -a "$RESULT"
else
  DENSE="${M%%|*}"; EXP="${M##*|}"
  PORT=$(python -c "import yanah_runtime as r; print(r.find_free_port(18898))")
  BUDGET=$(python -c "import coder_runtime as c; print(c.default_budget_gb(c.total_ram_gb()) or 10)")
  echo "    $DENSE + $EXP, --budget-gb $BUDGET, port $PORT"
  "$BIN" --role engine --port "$PORT" --model "$DENSE" --experts "$EXP" --budget-gb "$BUDGET" \
    --max-context 32768 --model-id qwen3-coder-next \
    --system-prompt-file "$APP/Contents/Frameworks/coder_system.md" > build/coder_smoke_engine.log 2>&1 &
  EPID=$!
  up=""
  for i in $(seq 1 600); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")
    [ "$code" = "200" ] && { up=1; break; }
    [ "$code" = "500" ] && break
    kill -0 $EPID 2>/dev/null || break
    sleep 1
  done
  if [ -n "$up" ]; then
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d '{
      "messages":[{"role":"user","content":"Read the file main.py using the read tool."}],
      "tools":[{"type":"function","function":{"name":"read","description":"Read a file",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],
      "max_tokens":120,"temperature":0}' > build/coder_smoke_reply.json
    python - <<'PY' | tee -a "$RESULT"
import json
r = json.load(open("build/coder_smoke_reply.json"))
c = r["choices"][0]
calls = c["message"].get("tool_calls") or []
if calls:
    f = calls[0]["function"]
    print(f"OK   smoke test: the model called {f['name']}({f['arguments']}) (finish_reason={c['finish_reason']})")
else:
    print(f"FAIL smoke test: no tool call came back; the model said: {(c['message'].get('content') or '')[:120]!r}")
PY
  else
    tail -20 build/coder_smoke_engine.log
    kill $EPID 2>/dev/null
    die "the built app's engine did not come up (build/coder_smoke_engine.log)"
  fi
  kill $EPID 2>/dev/null; wait $EPID 2>/dev/null
fi

step "dmg"
./make_coder_dmg.sh || die "dmg failed"
ok "dmg: $(ls dist/YanahCoder-*.dmg) ($(du -h dist/YanahCoder-*.dmg | cut -f1))"

echo; echo "Done."; cat "$RESULT"
echo "DONE" >> "$RESULT"
read -n1 -t 600 -p "Press any key to close."
