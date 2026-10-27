#!/bin/bash
# Package dist/Yanah.app into dist/Yanah-<version>.dmg (drag-to-Applications).
#
#   ./make_dmg.sh
#   SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" NOTARY_PROFILE=yanah ./make_dmg.sh
#
# Without SIGN_IDENTITY the app is ad-hoc signed: it runs on this Mac, and on
# other Macs only after the user allows it (System Settings > Privacy &
# Security > Open Anyway). With a Developer ID + notarization it opens normally.
# NOTARY_PROFILE is a keychain profile made once with:
#   xcrun notarytool store-credentials yanah --apple-id you@x.com --team-id TEAMID
set -euo pipefail
cd "$(dirname "$0")"

APP="dist/Yanah.app"
[ -d "$APP" ] || { echo "no $APP: run build_app.command first"; exit 1; }
VER=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist")
DMG="dist/Yanah-$VER.dmg"
STAGE="build/dmg"

if [ -n "${SIGN_IDENTITY:-}" ]; then
  echo "==> signing app with: $SIGN_IDENTITY"
  codesign --force --deep --options runtime --timestamp \
    --entitlements entitlements.plist --sign "$SIGN_IDENTITY" "$APP"
fi
codesign --verify --deep --strict "$APP" && echo "    app signature OK"

echo "==> staging"
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/Yanah.app"          # ditto keeps symlinks, xattrs and signatures intact
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/READ ME FIRST.txt" <<TXT
Drag Yanah into Applications, then open it from there.

The first time, Yanah downloads her model (about 30 GB). The download can be
paused and picks up where it left off.

Needs: a Mac with Apple Silicon (M1 or later), macOS 14 or later, and about
35 GB of free disk space. Works best with 16 GB of memory or more; on 8 GB
Macs she runs, but slowly, and works best with other apps closed.

If macOS says Yanah "can't be opened": open System Settings > Privacy &
Security, scroll down and click "Open Anyway".
TXT

echo "==> creating $DMG"
hdiutil create -volname "Yanah" -srcfolder "$STAGE" -ov -format UDZO -imagekey zlib-level=9 "$DMG" >/dev/null

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
