# Project Plan: Motorbike Pothole Detection & Telemetry System

## Strict 8-Stage Gated Engineering Pipeline
1. **Branch & Isolate**: Every task has its own feature branch created off `main` (`feat/<task-name>`).
2. **Contract & Interface First**: Define schemas, types, and function signatures.
3. **Red Test Suite**: Write failing unit and integration tests before implementation.
4. **Implement Minimal Code**: Write clean, modular, production-ready code to satisfy test requirements.
5. **Green Verification**: Run the automated test suite locally and verify 100% pass (0 failures).
6. **Lint & Static Analysis**: Ensure zero lint errors, type mismatches, or compiler warnings.
7. **Document & Status**: Record commit hashes, test results, and status in milestone document.
8. **Squash/Clean Merge to Main**: Merge branch into `main` after local verification.

---

## Phase Breakdown

### Phase 1: Rust Edge Vision & Geometry Core (`pothole_core`) - COMPLETED
- **Task 1.1**: Perspective distance math, lane classification, and NMS (`feat/task-1-1-rust-geometry-nms`).
- **Task 1.2**: Zero-copy ROI crop, float normalization, and IMU severity scoring (`feat/task-1-2-rust-preprocess-severity`).
- **Task 1.3**: YOLOv8 transposed `[1, 5, N]` tensor parser, export script (`feat/task-1-3-yolo-tensor-bridge`).

### Phase 2: Flutter HUD, Rust FFI & Offline Queue (`mobile_app`) - COMPLETED
- **Task 2.1**: C-ABI FFI bindings (`pothole_core/src/ffi.rs`) and Dart FFI bridge (`mobile_app/lib/rust_bridge.dart`).
- **Task 2.2**: SQLite offline hazard queue (`LocalHazardQueue`), 3.0m spatial deduplication, 50-item batch sync (`mobile_app/lib/local_db.dart`).
- **Task 2.3**: Tactical HUD overlay (`MotorbikePotholeHudApp`), <8m critical alert banner, and camera frame integration (`feat/task-2-3-flutter-hud-camera`).

### Phase 3: Python FastAPI Telemetry & 3-Meter Spatial Clustering Backend (`pothole-sync-backend`) - IN PROGRESS
- **Task 3.1 (`feat/task-3-1-fastapi-schemas-health`)**:
  - Backend project scaffolding (`pothole-sync-backend/`).
  - Pydantic v2 schemas: `SyncBatchRequest`, `PotholeDetectionItem` (enforcing $\le 100$ KB decoded `image_base64` thumbnails), `SyncResponse`.
  - `GET /health` endpoint returning system status.
  - Tests `T-3.1.1`, `T-3.1.2`, `T-3.1.3`.
- **Task 3.2 (`feat/task-3-2-spatial-clustering-db`)**:
  - SQLAlchemy models (`pothole_events`, `clustered_hazards`).
  - 3.0-meter Haversine spatial clustering algorithm (`CLUSTER_RADIUS_METERS = 3.0`).
  - `event_uuid` idempotency check ensuring no duplicate hazard records or double-counted clusters.
  - Tests `T-3.2.1`, `T-3.2.2`, `T-3.2.3`.
- **Task 3.3 (`feat/task-3-3-sync-nearby-gemini-triage`)**:
  - `POST /api/v1/telemetry/sync` endpoint accepting batches and updating clusters.
  - `GET /api/v1/hazards/nearby` endpoint querying clustered hazards within radius.
  - Async Gemini 2.5 Flash (`google-genai`) thumbnail false-positive triage worker.
  - Tests `T-3.3.1`, `T-3.3.2`, `T-3.3.3`.
  - Milestone status report: `STATUS-PHASE-3.md`.
