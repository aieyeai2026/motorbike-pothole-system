# TEST_COMPARISON — Old (`pre-phase4`) vs New (Phase 4 & Phase 5)

**Target:** $\ge 20\text{ FPS}$

## 1. Physical Phone Release Build Comparison

| metric | old | new | % change | faster/slower |
| :--- | :--- | :--- | :--- | :--- |
| FPS | N/A | N/A | N/A | N/A (awaiting real model file on phone) |
| Frame-to-HUD latency | N/A | N/A | N/A | N/A |
| Inference time | N/A | N/A | N/A | N/A |
| NMS time | N/A | N/A | N/A | N/A |
| Distance/lane math time | N/A | N/A | N/A | N/A |
| Dropped frames | N/A | N/A | N/A | N/A |
| Pothole alert delay | N/A | N/A | N/A | N/A |
| Pothole accuracy | N/A | N/A | N/A | N/A |
| SQLite insert time | N/A | N/A | N/A | N/A |
| CPU | N/A | N/A | N/A | N/A |
| Memory | N/A | N/A | N/A | N/A |

## 2. Host C-ABI + Flutter FFI Pipeline Comparison (p50 across 3 runs)

| metric | old | new | % change | faster/slower |
| :--- | :--- | :--- | :--- | :--- |
| FPS | 1020.40 FPS | 1020.41 FPS | +0.0% | unchanged (>= 20 FPS target met in FFI core) |
| Frame-to-HUD latency | 0.626 ms | 0.685 ms | +9.4% | slower (+0.059 ms for multi-class + `hazard_type`) |
| Inference time (Tensor Parse) | 0.018 ms | 0.009 ms | -50.0% | faster |
| NMS time | 0.006 ms | 0.002 ms | -66.7% | faster (per-class partition) |
| Distance/lane math time | 0.001 ms | 0.001 ms | 0.0% | unchanged |
| Dropped frames | 0 | 0 | 0.0% | unchanged |
| Pothole alert delay | 0.001 ms | 0.001 ms | 0.0% | unchanged |
| Pothole accuracy | N/A | N/A | N/A | N/A |
| SQLite insert time | 0.610 ms | 0.650 ms | +6.6% | slower (+0.040 ms for `hazard_type` column) |
| CPU | 19.0% | 19.0% | 0.0% | unchanged |
| Memory | 15.8 MB | 15.8 MB | 0.0% | unchanged |
