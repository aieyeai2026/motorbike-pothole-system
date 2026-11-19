# STATUS — Phase 2: Flutter HUD, Rust FFI & Offline Queue (`mobile_app`)
Date: 2026-11-19

## 1. PRs in Merge Order
1. PR #4 (`feat/task-2-2-sqlite-offline-queue` -> `main`, merge commit `616286d`): Offline SQLite queue (`LocalHazardQueue`), 3.0-meter Haversine deduplication, and 50-item batch flush (`T-2.2.1`..`T-2.2.3`).
2. PR #5 (`feat/task-2-3-flutter-hud-camera` -> `main`): Rust C-ABI FFI export (`pothole_core_run_postprocess`), Dart FFI bridge (`RustPotholeBridge`), Tactical HUD overlay with <8m critical alerts, and camera integration (`T-2.1.1`..`T-2.1.2`, `T-2.3.1`).

## 2. Test Counts (Quoted)
- `cd pothole_core && cargo test`:
  `"test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"`
- `cd mobile_app && flutter test`:
  `"00:00 +5: All tests passed!"` (5/5 tests: `rust_bridge_test.dart`, `local_db_test.dart` [3], `hud_widget_test.dart`).

## 3. CI Result
- `NOT CHECKED` — GitHub Actions CI workflow (`.github/workflows/ci.yml`) is scheduled for Phase 3.

## 4. What Is Not Proven Yet
- Real-time live camera streaming inference on physical Android hardware with TensorFlow Lite / ONNX Runtime dynamic library linking.
- Live HTTP sync from Android client to production FastAPI backend deployed in the cloud.

## 5. To Do By Hand
- Verify live camera permissions, 30 FPS HUD overlay rendering, and audio vibration alerts on physical motorbike mount.

## 6. Not Built Yet
- FastAPI backend (`pothole-sync-backend/`): Pydantic schemas, 3.0-meter spatial clustering DB, sync endpoint, nearby hazard query, and Gemini 2.5 Flash false-positive triage.

## 7. Open Decisions
- Standardized on 3.0-meter spatial clustering radius across both edge SQLite (`LocalHazardQueue`) and server PostgreSQL/SQLite deduplication to ensure consistent spatial hashing.
