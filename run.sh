#!/usr/bin/env bash
# ONE command.
#   curl -sL https://raw.githubusercontent.com/Yusasif-A/Helpa/main/run.sh | bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
say(){ printf "\n\033[1;32m▸ %s\033[0m\n" "$1"; }

if [[ ! -f app.py ]]; then
  say "getting the app..."
  command -v git >/dev/null || { echo "git is required — install it, then re-run"; exit 1; }
  git clone --depth 1 https://github.com/Yusasif-A/Helpa.git /tmp/helpa-src
  cd /tmp/helpa-src
fi

PY="$(command -v python3 || command -v python || true)"
[[ -n "$PY" ]] || { echo "Python 3.9+ required — https://python.org"; exit 1; }
say "using $($PY --version)"

if ! "$PY" -c "import gradio, requests, PIL, cv2" 2>/dev/null; then
  say "installing dependencies..."
  "$PY" -m pip install --quiet -r requirements.txt
fi

say "starting Helpa..."
echo "  First run downloads the model (~3.2 GB) and vision projector (~940 MB) once."
echo "  Then opens at http://localhost:7861 — runs on THIS computer from then on."
echo "  Ctrl+C to stop."
exec "$PY" app.py
