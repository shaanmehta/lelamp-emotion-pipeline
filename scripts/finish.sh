#!/usr/bin/env bash
# Sequential post-extraction pipeline.
# Deliberately sequential: two processes using Metal/MPS at once reliably wedges
# one of them on this machine (see README, Limitations).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src

echo "[finish] waiting for clip_test.npz ..."
until [ -f artifacts/features/clip_test.npz ]; do sleep 15; done
sleep 5
echo "[finish] vision features present:"; ls -la artifacts/features/

echo "[finish] 1/4 text features"
$PY -u scripts/extract_text.py 2>&1 | grep -E "^\[text\]"

echo "[finish] 2/4 training heads"
$PY -u -m lelamp.train 2>&1 | grep -E "^\[(train|data)\]"

echo "[finish] 3/4 evaluation"
$PY -u -m lelamp.evaluate.run_all 2>&1 | grep -vE "Loading weights|Fetching"

echo "[finish] 4/4 README results"
$PY scripts/render_results.py
echo "[finish] DONE"
