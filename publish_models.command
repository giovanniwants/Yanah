#!/bin/bash
# Publish Yanah's model files to your Hugging Face account, then point the
# app build at them. Double-click. Takes a while (packages ~30 GB, then uploads it).
#
# Your token never leaves this Terminal: if you aren't logged in, `hf auth login`
# asks for it here, with the input hidden, and stores it in ~/.cache/huggingface.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "no .venv here"; read -n1; exit 1; }
mkdir -p build
exec > >(tee build/publish.log) 2>&1

echo "==> Hugging Face login"
if ! hf auth whoami >/dev/null 2>&1; then
  echo "    Not logged in. Paste a token with WRITE access"
  echo "    (huggingface.co > Settings > Access Tokens > Create new token > Write)."
  hf auth login || { echo "login failed"; read -n1 -p "Press any key."; exit 1; }
fi
USER_NAME=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])") || {
  echo "could not read your account (expired token?). Run: hf auth logout, then this again."; read -n1; exit 1; }
echo "    logged in as $USER_NAME"
# whoami() reports the token's role; uploading needs write (or fine-grained with repo write)
ROLE=$(python -c "from huggingface_hub import whoami; a=whoami().get('auth',{}).get('accessToken',{}); print(a.get('role','?'))")
if [ "$ROLE" = "read" ]; then
  echo "    This token is READ-only, so it can't upload. Make a Write token, then:"
  echo "      hf auth login     (paste the new token)"
  read -n1 -p "Press any key."; exit 1
fi

DEFAULT_REPO="$USER_NAME/Yanah-Qwen3-Next-80B"
REPO="$(cat model_repo.txt 2>/dev/null | tr -d '[:space:]')"
REPO="${REPO:-$DEFAULT_REPO}"
echo
echo "==> Will publish PUBLICLY to https://huggingface.co/$REPO"
echo "    (about 30 GB: Qwen3-Next dense weights + 2-bit experts, Apache-2.0 with its LICENSE)"
read -p "    Type yes to continue: " OK
[ "$OK" = "yes" ] || { echo "cancelled"; read -n1 -p "Press any key."; exit 0; }

echo
python publish_models.py --repo "$REPO" --upload | tee build/published_repo.txt
STATUS=${PIPESTATUS[0]}
if [ "$STATUS" -ne 0 ]; then
  echo; echo "Publishing stopped (see build/publish.log). Run this again to resume: finished files are skipped."
  read -n1 -p "Press any key."; exit 1
fi
FINAL=$(tail -1 build/published_repo.txt | tr -d '[:space:]')
echo "$FINAL" > model_repo.txt
echo
echo "Published: https://huggingface.co/$FINAL"
echo "Saved to model_repo.txt. Run build_app.command again so the app downloads from it."
read -n1 -p "Press any key to close."
