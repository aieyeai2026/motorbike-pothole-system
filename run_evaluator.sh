#!/usr/bin/env bash
# One-shot launcher for the video evaluator (Local & Cloud).
# Usage:  bash run_evaluator.sh
#
# GOOGLE CLOUD DEPLOYMENT (Live Link on Port 8085):
# 1. Cloud Shell Web Preview:
#    - Upload deploy_package.zip or clone repo
#    - Run: bash run_evaluator.sh
#    - Click Web Preview icon -> "Preview on port 8085" -> Instant Live URL
# 2. Google Cloud Run (Permanent URL):
#    - gcloud run deploy pothole-evaluator --source . --port 8085 --memory 2Gi --cpu 2 --allow-unauthenticated
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8085}"
HOST="${HOST:-0.0.0.0}"

# 1. stop any old evaluator still holding the port (this is what keeps serving the stale page)
if lsof -ti tcp:$PORT >/dev/null 2>&1; then
  echo "Stopping old server on :$PORT"
  lsof -ti tcp:$PORT | xargs kill 2>/dev/null || true
  sleep 1
fi

# 2. dependencies
python3 - <<'PY' || { echo "Installing missing packages..."; python3 -m pip install torch opencv-python numpy; }
import torch, cv2, numpy  # noqa
PY

# 3. required files
[ -f user_test_experience/video_evaluator/models/pyresearch_pothole_best.pt ] || echo "NOTE: weights missing; they will be downloaded from the pyresearch repo on first run"
if [ ! -f pothole_core/target/release/libpothole_core.dylib ] && [ ! -f pothole_core/target/release/libpothole_core.so ]; then
  echo "Rust shared library missing -> building with cargo"
  cargo build --release --manifest-path pothole_core/Cargo.toml
fi

# 4. run (open browser if running in GUI desktop environment)
if command -v open >/dev/null 2>&1; then
  ( sleep 3 && open "http://127.0.0.1:$PORT/" ) >/dev/null 2>&1 &
fi

echo "Evaluator on http://${HOST}:${PORT}/ (Ctrl+C to stop)."
exec python3 user_test_experience/video_evaluator/server.py --host "$HOST" --port "$PORT"
