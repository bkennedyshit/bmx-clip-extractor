#!/usr/bin/env bash
set -euo pipefail

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r workspace/requirements.txt

echo
echo "[done] Setup complete."
echo "Run: venv/bin/python workspace/bmx_riding_extractor.py --list-models"
