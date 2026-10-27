"""
Adapts the bundled OpenCode desktop app (OpenCode.app, Electron) for YanahCoder.
build_coder.command runs it on the copy inside YanahCoder.app, then re-signs that copy.

Three edits to OpenCode's startup code (app.asar: out/main/index.js), each the same
length as the text it replaces, so nothing else in the archive moves:

  1. App ID "ai.opencode.desktop" -> "app.yanah.coder.app" (APP_IDS.prod). OpenCode keeps its
     tabs and settings, and its one-copy-at-a-time lock, in a folder named after this ID. With
     its own ID, YanahCoder's OpenCode runs alongside a separately installed OpenCode instead
     of handing over to it, and never shows that one's tabs.
  2. The auto-updater is switched off (UPDATER_ENABLED = false): an in-place update would
     replace the OpenCode inside YanahCoder.app and break the app. New OpenCode versions come
     with new YanahCoder builds.
  3. It no longer makes itself the system's handler for opencode:// links (which would take
     them away from a separately installed OpenCode). YanahCoder still opens it with such a
     link directly (`open -a <this app> opencode://...`), which doesn't need the registration.

Electron checks app.asar against a SHA-256 of its header (Info.plist ElectronAsarIntegrity),
and each file against hashes stored in that header. Both are recomputed here, and
CFBundleIdentifier in Info.plist is set to the new ID.

    python patch_opencode.py path/to/OpenCode.app
"""
from __future__ import annotations

import hashlib
import json
import plistlib
import struct
import sys
from pathlib import Path

NEW_ID = "app.yanah.coder.app"
FILE = ("out", "main", "index.js")

EDITS = [   # (what OpenCode 1.19.x ships, what YanahCoder needs); padded to the same length
    ('  prod: "ai.opencode.desktop"\n};\nconst TEST_ONBOARDING',
     f'  prod: "{NEW_ID}"\n}};\nconst TEST_ONBOARDING'),
    ('const UPDATER_ENABLED = app.isPackaged && CHANNEL !== "dev";',
     'const UPDATER_ENABLED = false;'),
    ('app.setAsDefaultProtocolClient("opencode");',
     'void 0;'),
]


def _read_header(asar: Path):
    with open(asar, "rb") as f:
        size_pickle_len, header_size, payload_len, json_len = struct.unpack("<IIII", f.read(16))
        header_json = f.read(json_len)
    return header_size, header_json


def _entry(header: dict, path) -> dict:
    node = header
    for part in path:
        node = node["files"][part]
    return node


def patch(app: Path) -> dict:
    asar = app / "Contents" / "Resources" / "app.asar"
    plist_path = app / "Contents" / "Info.plist"
    header_size, header_json = _read_header(asar)
    header = json.loads(header_json)
    entry = _entry(header, FILE)
    data_start = 8 + header_size
    offset, size = data_start + int(entry["offset"]), entry["size"]
    with open(asar, "rb") as f:
        f.seek(offset)
        original = f.read(size)
    if hashlib.sha256(original).hexdigest() != entry["integrity"]["hash"]:
        raise SystemExit("app.asar: out/main/index.js doesn't match its recorded hash; not patching")

    text = original.decode("utf-8")
    done = []
    for old, new in EDITS:
        new = new.ljust(len(old))                      # same length: offsets stay valid
        if len(new) != len(old):
            raise SystemExit(f"edit longer than the original: {old!r}")
        if text.count(old) == 1:
            text = text.replace(old, new)
            done.append(old.split("\n")[0][:50])
        elif text.count(new) == 1:
            done.append("(already) " + old.split("\n")[0][:40])
        else:
            raise SystemExit(f"OpenCode's code changed: expected exactly one {old[:60]!r}. "
                             "Check the edits in patch_opencode.py against this OpenCode version.")
    patched = text.encode("utf-8")
    assert len(patched) == len(original)

    # new hashes for the file (one block per blockSize)
    bs = entry["integrity"].get("blockSize", 4 * 1024 * 1024)
    new_hash = hashlib.sha256(patched).hexdigest()
    new_blocks = [hashlib.sha256(patched[i:i + bs]).hexdigest() for i in range(0, len(patched), bs)]
    old_hashes = [entry["integrity"]["hash"]] + entry["integrity"]["blocks"]
    new_hashes = [new_hash] + new_blocks
    header_text = header_json.decode("utf-8")
    key = f'"size":{size},"offset":"{entry["offset"]}"'      # this file's entry (offsets are unique)
    if header_text.count(key) != 1:
        raise SystemExit("app.asar header: can't find index.js's entry")
    start = header_text.index(key)
    end = header_text.index("]}", start) + 2                   # end of its "blocks":[...]}
    segment = header_text[start:end]
    for old_h, new_h in dict(zip(old_hashes, new_hashes)).items():   # a 1-block file: block hash == file hash
        if old_h not in segment:
            raise SystemExit("app.asar header: hash not found in index.js's entry")
        segment = segment.replace(old_h, new_h)               # same length (hex)
    header_text = header_text[:start] + segment + header_text[end:]
    new_header = header_text.encode("utf-8")
    assert len(new_header) == len(header_json)

    with open(asar, "r+b") as f:
        f.seek(offset)
        f.write(patched)
        f.seek(16)
        f.write(new_header)

    info = plistlib.loads(plist_path.read_bytes())
    info["CFBundleIdentifier"] = NEW_ID
    info["ElectronAsarIntegrity"]["Resources/app.asar"]["hash"] = hashlib.sha256(new_header).hexdigest()
    plist_path.write_bytes(plistlib.dumps(info))
    return {"edits": done, "bundle_id": NEW_ID}


def verify(app: Path) -> None:
    """Re-reads everything the way Electron checks it."""
    asar = app / "Contents" / "Resources" / "app.asar"
    header_size, header_json = _read_header(asar)
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    want = info["ElectronAsarIntegrity"]["Resources/app.asar"]["hash"]
    assert hashlib.sha256(header_json).hexdigest() == want, "header hash != Info.plist"
    entry = _entry(json.loads(header_json), FILE)
    with open(asar, "rb") as f:
        f.seek(8 + header_size + int(entry["offset"]))
        body = f.read(entry["size"])
    assert hashlib.sha256(body).hexdigest() == entry["integrity"]["hash"], "file hash != header"
    text = body.decode("utf-8")
    assert f'prod: "{NEW_ID}"' in text and "const UPDATER_ENABLED = false;" in text
    assert 'app.setAsDefaultProtocolClient("opencode")' not in text
    assert info["CFBundleIdentifier"] == NEW_ID


if __name__ == "__main__":
    target = Path(sys.argv[1])
    print(json.dumps(patch(target)))
    verify(target)
    print("verified: app.asar integrity, Info.plist hash, bundle id")
