#!/bin/bash
# Package dist/YanahCoder.app into dist/YanahCoder-<version>.dmg (drag-to-Applications).
# build_coder.command signs the app before this runs (OpenCode.app keeps its own signature,
# so this script never re-signs the app with --deep).
#
#   ./make_coder_dmg.sh
#   SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" NOTARY_PROFILE=yanah ./make_coder_dmg.sh
set -euo pipefail
cd "$(dirname "$0")"

APP="dist/YanahCoder.app"
[ -d "$APP" ] || { echo "no $APP: run build_coder.command first"; exit 1; }
[ -d "$APP/Contents/Resources/OpenCode.app" ] || { echo "OpenCode.app missing from $APP: run build_coder.command"; exit 1; }
VER=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist")
DMG="dist/YanahCoder-$VER.dmg"
STAGE="build/dmg_coder"

codesign --verify --strict "$APP" && echo "    app signature OK"

echo "==> staging"
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/YanahCoder.app"
ln -s /Applications "$STAGE/Applications"
cp third_party/OPENCODE_LICENSE.txt "$STAGE/OpenCode license (MIT).txt"
cat > "$STAGE/READ ME FIRST.txt" <<TXT
Yanah Coder: a local coding agent. The OpenCode app talks to Qwen3-Coder-Next
running on this Mac. No account, no cloud, no API key.

Drag YanahCoder into Applications, then open it from there. It starts the
engine and opens OpenCode in a new chat, already connected, working in
~/YanahCoder. Open any other project from OpenCode's Home screen.

YanahCoder brings its own copy of OpenCode, which runs next to an OpenCode
you may already have installed, and doesn't update itself: new OpenCode
versions come with new YanahCoder releases.

Needs: a Mac with Apple Silicon (M1 or later; M3 or later recommended),
macOS 14 or later, at least 16 GB of memory, and about 50 GB of free disk
space for the model.

The model: if this build offers a download, YanahCoder fetches it on first
launch (about 45 GB, resumable). Otherwise prepare it once with
prepare_coder_models.command from the source folder.

If macOS says YanahCoder "can't be opened": open System Settings > Privacy &
Security, scroll down and click "Open Anyway" (once). From Terminal instead:
    xattr -dr com.apple.quarantine /Applications/YanahCoder.app
TXT

echo "==> creating $DMG"
hdiutil create -volname "Yanah Coder" -srcfolder "$STAGE" -ov -format UDZO -imagekey zlib-level=9 "$DMG" >/dev/null

if [ -n "${SIGN_IDENTITY:-}" ]; then
  codesign --sign "$SIGN_IDENTITY" --timestamp "$DMG"
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    echo "==> notarizing (a few minutes)"
    xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG"
  fi
fi
hdiutil verify "$DMG" >/dev/null && echo "    dmg verified"
echo "==> $DMG ($(du -h "$DMG" | cut -f1))"
