#!/usr/bin/env bash
# One-shot launcher for the video evaluator. Usage:  bash run_evaluator.sh
set -e
cd "$(dirname "$0")"
PORT=8085

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
[ -f pothole_core/target/release/libpothole_core.dylib ] || { echo "Rust lib missing -> building"; cargo build --release --manifest-path pothole_core/Cargo.toml; }

# 4. run (open the browser a moment after the server starts)
( sleep 3 && open "http://127.0.0.1:$PORT/" ) >/dev/null 2>&1 &
echo "Evaluator on http://127.0.0.1:$PORT/  (Ctrl+C to stop). Re-upload the video: old processed output used the old threshold."
exec python3 user_test_experience/video_evaluator/server.py
