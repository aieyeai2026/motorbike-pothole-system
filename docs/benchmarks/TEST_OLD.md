# TEST_OLD — Pre-Phase 4 Baseline

**Git tag:** `pre-phase4` (commit `d30fc0e`), no app code changed.
**Status:** **No phone test run till now.**
**Fixtures:** `benchmarks/fixtures/ride_test.mp4` (848×478, 59.5 FPS, 02:29), added 2026-11-20. Not yet run through a phone build.

## Why everything is N/A
1. No release build has been run on the iPhone yet.
2. The current app has **no detection model**. `main.dart` runs `_simMode = true` and feeds a hard-coded tensor to Rust, so camera frames never reach inference. FPS, latency and accuracy on real footage cannot be measured on this architecture.
3. No ground-truth labels exist for the fixture video, so accuracy cannot be scored.

## Metrics (phone, release build, 3 runs)

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

## Host-only reference (macOS, mock tensors — NOT phone numbers)
Measured on the Mac with `cargo test --release` / `flutter test`. Useful only to show the Rust post-processing is not a bottleneck.

| Step | Value |
| :--- | :--- |
| YOLO tensor parse, 8400 anchors | 0.018 ms |
| NMS | 0.006 ms |
| Distance + lane math | 0.001 ms |
| ROI crop + normalise, 640×480 | 1.16 ms |
| SQLite insert (in-memory) | ~0.61 ms p50 |

## Likely bottlenecks
1. Model inference (not present yet), which will dominate frame time.
2. `crop_and_normalize_roi` runs on the CPU (~1.2 ms on the Mac; the phone will be slower).
3. SQLite writes on the UI isolate in `_startSimulationLoop`.

## How to re-run
```bash
python3 benchmarks/run_benchmark.py            # host mock run -> docs/benchmarks/TEST_HOST_MOCK.md
cargo test --manifest-path pothole_core/Cargo.toml --release --test bench_metrics -- --nocapture
cd mobile_app && flutter test test/pipeline_benchmark_test.dart
```
To get real phone numbers you need a model file and `flutter run --release` on the iPhone.
