#!/usr/bin/env bash
# Launch the Daft Punk Vocal DSP web app.
set -e
cd "$(dirname "$0")"

# Optional but recommended: espeak-ng gives real speech; without it the engine
# falls back to a built-in vowel-babble synth.
if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "note: espeak-ng not found (using fallback synth)."
  echo "      install with:  sudo apt-get install -y espeak-ng"
fi

python3 -c "import numpy, flask" 2>/dev/null || {
  echo "installing python deps..."
  python3 -m pip install -r requirements.txt
}

export PORT="${PORT:-8000}"
echo "starting on http://0.0.0.0:${PORT}"
exec python3 server/app.py
