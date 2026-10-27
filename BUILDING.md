# Building and shipping Yanah.app

## Build

Double-click `build_app.command` in this folder. It checks the toolchain,
runs the app tests, builds `dist/Yanah.app`, self-tests the built app (Metal
GPU, imports, tokenizer), runs the built app's engine on your real models,
and makes `dist/Yanah-0.1.0.dmg`. The summary is in `build/BUILD_RESULT.txt`.

## Let other Macs download the model (once)

The app downloads from a Hugging Face repository you own:

1. `python publish_models.py` builds `~/Desktop/yanah_publish` (about 30 GB):
   the slim dense weights (1.3 GiB), 48 expert parts and a manifest of sha256s.
2. Make a free Hugging Face account and a write token, then run `hf auth login`.
3. `python publish_models.py --repo YOURNAME/Yanah-Qwen3-Next-80B --upload`.
   If the upload stops, run it again: it resumes.
4. Put `YOURNAME/Yanah-Qwen3-Next-80B` in `model_repo.txt` and rebuild.

The model is Apache-2.0 (Qwen3-Next-80B-A3B-Instruct), so redistributing the
quantized weights is allowed. The model card credits it.

## Signing (for Macs other than yours)

Without a Developer ID the app is ad-hoc signed. Other Macs block it until
the user clicks **Open Anyway** in System Settings > Privacy & Security. To
avoid that you need an Apple Developer account ($99/year):

    SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" NOTARY_PROFILE=yanah ./make_dmg.sh

## Where things go on the user's Mac

| what | where |
|---|---|
| app | /Applications/Yanah.app (read-only) |
| models | ~/Library/Application Support/Yanah/models (or an existing ~/Desktop/yanah_models) |
| chats, memories, editable identity files, settings.json | ~/Library/Application Support/Yanah |
| logs (app, engine, shell) | ~/Library/Application Support/Yanah/logs |

`settings.json` (optional) overrides: `budget_gb`, `max_context`, `experts`
(`auto`/`2bit`/`4bit`), `model_repo`, `engine_port`, `shell_port`.

## How each failure is handled

| failure | what happens |
|---|---|
| no or broken Python on the user's Mac | the app carries its own Python and every library; it never uses the system's |
| Wi-Fi drops mid-download | retried forever with backoff (2 s up to 60 s); the file in flight resumes from its last byte; finished parts are kept |
| a file arrives corrupted | each file is checked against the sha256 the repository records; a bad one is downloaded again (only that 576 MB part) |
| quit / sleep / crash during download | progress saved per part; Start Download resumes |
| disk full | checked before starting (needs the download + 5 GB spare); plain message |
| models missing on first launch | the window offers the download instead of starting the engine |
| port 8888 / 8901 in use | the next free port is used; the chat page is served by the shell, so it always finds it |
| engine killed for memory (or crashes) | the chat says "Yanah's engine has paused…", the window shows why, the engine restarts (5 s, 20 s, 45 s); after 3 crashes in 10 min it waits for **Restart Engine** |
| engine hangs | the shell gives up after 45 s without data (the engine sends a keepalive every 5 s) |
| model files damaged | engine load error is shown with **Check Files**, which re-hashes everything and re-downloads only the damaged parts |
| launcher force-quit | engine and shell notice within 2 s and exit, so no memory stays held |
