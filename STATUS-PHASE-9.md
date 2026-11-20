# Phase 9 Milestone Status: Pure PyResearch Neural Detector (>= 0.90 Threshold) & Modular 6-Layer AR HUD Toggles

**Branch**: `feat/phase9-pyresearch-pothole-80pct` -> `main`  
**Status**: **COMPLETE — 100% Test Pass Rate & Unanimous Dual-Subagent Audit PASS**

---

## 1. Objectives Delivered

### A. Pure PyResearch YOLOv12s/YOLOv8 Neural Detector (`>= 0.90` Strict Confidence Gate)
- Integrated trained weights from [`pyresearch/Pothole-Computer-Vision-Project`](https://github.com/pyresearch/Pothole-Computer-Vision-Project) (`best.pt`, `nc=1`, `{0: 'Pothole'}`, cached locally as ignored `*.pt` at `user_test_experience/video_evaluator/models/pyresearch_pothole_best.pt`) with Apple Silicon MPS / CPU batched PyTorch inference.
- Removed all heuristic morphological BlackHat, Gaussian background subtraction, and Sobel gradient contour generation logic from [`pyresearch_pothole_detector.py`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/pyresearch_pothole_detector.py).
- Runs batched neural inference strictly on **aspect-preserving `1:1` square (`140x140` px) road-corridor crops** (up to 3 darkest drivable-road seeds + 2 center-corridor crops) with `0.12 <= bw_c <= 0.70`, `0.10 <= bh_c <= 0.70`, and faithful unprojection (eliminating `4:1` tile stretching and box shrinkage).
- Enforces **`RAW_YOLO_GATE = 0.55`** (`DEFAULT_CONF_THRESHOLD = 0.90`), exterior `pad=10` `road_only_mask` asphalt verification (`>= 0.55`), sunlit road glare/sticker guards, widened bottom-center dome/windshield suppression (`y_max >= 0.68` in center `[0.25, 0.75]`), and honors `use_yolo=False` (`([], 0)`).

### B. Converging Rust-Parity Boundaries, Close Vehicle/Tanker Retention & Distance Ticks
- Synchronized [`_fit_3pt_physical_bounds`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/pyresearch_pothole_detector.py#L250-L330), [`geometry.rs`](file:///Users/gautam/motorbike-pothole-system/pothole_core/src/geometry.rs), and [`main.dart`](file:///Users/gautam/motorbike-pothole-system/mobile_app/lib/main.dart#L721) on `kappa_road * t * sin(pi * t)` converging cleanly at `(x_vp, y_vp)` with `y_vp` calibrated in `[0.48, 0.56]` and scan-outward lower road edges avoiding adjacent vehicles.
- Fixed [`extract_onroad_obstacles_for_carpet`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/server.py#L525-L607) to retain close tall vehicles/tankers extending to `y_max >= 0.86` (clamped to `0.85`), eliminating carpet-into-tanker overlap while keeping bottom-center helmet/windshield filtering intact.
- Added [`compute_perspective_distance_ticks`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/server.py#L842-L892) (`5m`, `10m`, `15m`, `20m`) clipped to `[y_target_norm, y_bot]` and 2-pass non-overlapping pothole box + pill rendering in `render_phase8_ar_hud`.

### C. Modular 6-Layer Perception & HUD Toggle Workbench (`http://127.0.0.1:8085/`)
- Added independent UI toggle switches and live `<canvas id="hud-overlay-canvas">` compositing in [`index.html`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/index.html), [`style.css`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/style.css), [`app.js`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/app.js), and [`server.py`](file:///Users/gautam/motorbike-pothole-system/user_test_experience/video_evaluator/server.py) for all 6 perception subsystems:
  1. `horizon` — **Tilting Green Horizon**
  2. `boundaries` — **Curved Cyan Road Boundaries**
  3. `carpet` — **Solid Green Trajectory Carpet**
  4. `potholes` — **PyResearch Pothole Boxes (>=90%)**
  5. `distance_ticks` — **Distance Ticks (5m–20m)**
  6. `telemetry` — **Telemetry Overlays**
- Added 3 quick-isolation buttons (`All Layers ON`, `Isolate Potholes Only (>=90%)`, `Clean Raw View`) and per-request HTTP header controls (`X-Hud-Layers` JSON + `X-Conf-Threshold` clamped `>= 0.90`).

---

## 2. Verification & Benchmark Results (Post-Final-Edit)

| Suite / Check | Result | Notes |
| :--- | :---: | :--- |
| **Rust Core (`cargo test --manifest-path pothole_core/Cargo.toml`)** | **18 passed, 0 failed + 1 bench ok** | `cargo build --release` clean (0 warnings) |
| **Flutter Mobile App (`flutter test` + `flutter analyze --no-fatal-infos`)** | **13 passed, 0 failed** | 0 errors/warnings (`5` pre-existing infos) |
| **Sync Backend (`pytest pothole-sync-backend/ -v`)** | **9 passed, 0 failed** | Clustering, schema, health endpoints |
| **UX & Video Evaluator (`pytest user_test_experience/tests/test_user_experience.py -v`)** | **10 passed, 0 failed (`15.79s`)** | `T-UX-1`..`T-UX-10` including HTTP header & 6-layer isolation assertions |
| **Random 1-Min Dual Benchmark (`[37.78s .. 97.78s]`, 12 random timestamps)** | **11/12 (`91.67%`) recall @ `96.43%` mean conf** | `smooth_road_fp_count = 0`, `windshield_fp_count = 0`, `offroad_fp_count = 0` |
| **9 Random Verification Frames (`benchmarks/fixtures/phase8_verification/frame_01..09.jpg`)** | **Visual PASS** | `Pot:0` false-alarm free on smooth `ride_test.mp4`, clean converging bounds & tanker weaving |
| **Subagent Audit (`500f4a06` & `669c9948`)** | **PASS / PASS** | Unanimous sign-off after remediation |
