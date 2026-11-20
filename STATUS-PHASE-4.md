# STATUS — Phase 4: Multi-Class Hazards, Road Edges & Biker Position
Date: 2026-11-20

## 1. Summary of Changes
- **Multi-Class Tensor Parsing & Per-Class NMS (`pothole_core/src/inference.rs`, `lib.rs`)**:
  - Extended YOLOv8 transposed tensor decoding from `[1, 5, N]` to `[1, 4+C, N]` (`0 = pothole`, `1 = car`, `2 = debris`) while preserving backwards compatibility for single-class tensors.
  - Added per-class NMS so vehicle bounding boxes never suppress road potholes directly ahead.
- **C-ABI FFI Exports (`pothole_core/src/ffi.rs`)**:
  - Extended `CDetectedPothole` with `class_id: i32`.
  - Added `CLaneCentering` struct and exported `pothole_core_compute_lane_centering` for road boundary offset and rider drift status (`CENTERED`, `DRIFTING_LEFT`, `DRIFTING_RIGHT`).
- **Flutter FFI Bridge & SQLite Queue (`mobile_app/lib/rust_bridge.dart`, `local_db.dart`, `main.dart`)**:
  - Updated `PotholeHazard` with `classId` and `hazardType` (`pothole`, `car`, `debris`) and added `RustPotholeBridge.computeLaneCentering`.
  - Upgraded `pending_potholes` SQLite schema to v2 with `hazard_type TEXT NOT NULL DEFAULT 'pothole'`.
  - Filtered out `'car'` detections from entering the sync queue (`enqueueIfNotDuplicate` returns `false` for `'car'`), leaving the backend API completely unchanged.
  - Scoped `< 8m` critical red alert banner strictly to potholes (`hazardType == 'pothole'`).

## 2. Test Counts (Quoted)
- `cargo test --manifest-path pothole_core/Cargo.toml`:
  `"test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"` (+1 integration benchmark test)
- `cd mobile_app && flutter test`:
  `"00:00 +9: All tests passed!"`
- `pytest pothole-sync-backend/ user_test_experience/tests/`:
  `"14 passed, 1 warning in 0.39s"`
