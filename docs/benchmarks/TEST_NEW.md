# TEST_NEW — Post-Phase 4 & Phase 5 Benchmark Report

**Branch:** `feat/phase5-hardware-safeguards-benchmarks`
**Fixtures:** `benchmarks/fixtures/ride_test.mp4` (848×478, 59.5 FPS, 02:29) + 22 extracted frames
**Status:** Host Rust C-ABI + Flutter FFI pipeline measured over 3 runs; on-device physical release build metrics remain `N/A` until a trained `.tflite`/CoreML model file is deployed to the phone.

## 1. Baseline Metrics (Same Metric Names as `TEST_OLD.md`)

### 1.1 Physical Phone Release Build (3 Runs)

| Metric | Run 1 | Run 2 | Run 3 | p50 | p95 | Unit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FPS | N/A | N/A | N/A | N/A | N/A | FPS |
| Frame-to-HUD latency | N/A | N/A | N/A | N/A | N/A | ms |
| Inference time | N/A | N/A | N/A | N/A | N/A | ms |
| NMS time | N/A | N/A | N/A | N/A | N/A | ms |
| Distance/lane math time | N/A | N/A | N/A | N/A | N/A | ms |
| Dropped frames | N/A | N/A | N/A | N/A | N/A | frames |
| Pothole alert delay | N/A | N/A | N/A | N/A | N/A | ms |
| Pothole accuracy | N/A | N/A | N/A | N/A | N/A | % |
| SQLite insert time | N/A | N/A | N/A | N/A | N/A | ms |
| CPU | N/A | N/A | N/A | N/A | N/A | % |
| Memory | N/A | N/A | N/A | N/A | N/A | MB |

### 1.2 Host Pipeline Measurements (macOS Release `libpothole_core.dylib` + Flutter FFI, 3 Runs)

| Metric | Run 1 | Run 2 | Run 3 | p50 | p95 | Unit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FPS | 980.39 | 1020.41 | 1020.41 | **1020.41** | **1020.41** | FPS |
| Frame-to-HUD latency | 0.685 | 0.693 | 0.670 | **0.685** | **0.693** | ms |
| Inference time (Tensor Parse) | 0.029 | 0.009 | 0.009 | **0.009** | **0.029** | ms |
| NMS time | 0.006 | 0.002 | 0.002 | **0.002** | **0.006** | ms |
| Distance/lane math time | 0.002 | 0.000 | 0.000 | **0.000** | **0.002** | ms |
| Dropped frames | 0 | 0 | 0 | **0** | **0** | frames |
| Pothole alert delay | 0.000 | 0.000 | 0.000 | **0.000** | **0.001** | ms |
| Pothole accuracy | N/A | N/A | N/A | **N/A** | **N/A** | % |
| SQLite insert time | 0.646 | 0.650 | 0.658 | **0.650** | **0.658** | ms |
| CPU | 19.0 | 19.1 | 19.0 | **19.0** | **19.1** | % |
| Memory | 15.8 | 15.8 | 15.8 | **15.8** | **15.8** | MB |

---

## 2. Phase 4 & Phase 5 New Metrics

| New Metric | Run 1 | Run 2 | Run 3 | p50 | p95 | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Road boundary math** | 0.001 | 0.001 | 0.001 | **0.001** | **0.001** | ms | `pothole_core_compute_lane_centering` + pitch horizon |
| **Biker position FFI** | 0.002 | 0.002 | 0.002 | **0.002** | **0.002** | ms | `RustPotholeBridge.computeLaneCentering` |
| **RTSP delay** | N/A | N/A | N/A | **N/A** | **N/A** | ms | No RTSP dashcam package installed |
| **Per-class accuracy (pothole / car / debris)** | N/A | N/A | N/A | **N/A** | **N/A** | % | Requires ground-truth labels & real model file |
