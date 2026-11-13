# STATUS — Phase 1: Rust Edge Vision & Geometry Core (`pothole_core`)
Date: 2026-11-13

## 1. PRs in Merge Order
1. PR #1 (`feat/task-1-1-rust-geometry-nms` -> `main`, merge commit `2b7387a`): Perspective distance math, lane classification, and NMS (`T-1.1.1`..`T-1.1.4`).
2. PR #2 (`feat/task-1-2-rust-preprocess-severity` -> `main`, merge commit `50119e2`): Zero-copy ROI crop, float normalization, and IMU severity scoring (`T-1.2.1`..`T-1.2.2`).
3. PR #3 (`feat/task-1-3-yolo-tensor-bridge` -> `main`): YOLOv8 transposed `[1, 5, N]` tensor parser, `export_yolov8n.py`, and `.gitignore` cleanup (`T-1.3.1`).

## 2. Test Counts (Quoted)
- `cargo test` (2026-11-13): `"test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"`

## 3. CI Result
- `NOT CHECKED` — GitHub Actions CI workflow (`.github/workflows/ci.yml`) is scheduled for Phase 3.

## 4. What Is Not Proven Yet
- Real-time camera frame streaming and FFI binding on a physical Android device.
- End-to-end SQLite offline queueing and batch sync to FastAPI.

## 5. To Do By Hand
- Run the Flutter + Rust FFI HUD on a physical Android device in Phase 2.

## 6. Not Built Yet
- Flutter mobile HUD app (`mobile_app/`) and FastAPI backend (`pothole-sync-backend/`).

## 7. Open Decisions
- Re-ordered Phase 2 and Phase 3 so the **Flutter + Rust FFI Android Mobile App** is built in **Phase 2**, allowing every subsequent phase to be tested directly on the physical Android device.
