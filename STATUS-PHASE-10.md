# Phase 10 Status: Honest YOLOv8 Pothole Gate (raw 0.45) & Measured Curved Road Boundaries

**Status**: Implemented and verified on `test2pothole.mp4`. Rust-dependent suites (T-UX-7/8/9) and the web UI still need a run on macOS (see §4).

## 1. Pothole detection (Phase 1)
- Weights: `pyresearch/Pothole-Computer-Vision-Project` `best.pt` (cached, git-ignored, `video_evaluator/models/pyresearch_pothole_best.pt`).
- **Gate is the raw YOLO confidence, default `0.45`, callers may only raise it.** The Phase 9 `calibrate_pyresearch_confidence` remap (raw 0.55 displayed as "0.90") is removed. The weights never exceed 0.81 on this video, so a literal 0.90 shows nothing; the user chose 0.45 after 0.75 proved too strict (4 boxes).
- Inference is plain full-frame YOLO on a 1280x736 letterbox. Removed: darkest-seed 140px crops, corridor/asphalt masks, glare/sticker/dome heuristics, calibrated benchmark with synthetic craters.
- Kept: greedy NMS, size sanity cap, skip boxes centred in a detected foreground vehicle, Rust NMS downstream.
- UI: toggle `Pothole Detection (YOLOv8 @ 0.45)`; `Isolate Potholes Only` button; request header `X-Conf-Threshold` clamped to `>= 0.75`.

## 2. Road boundaries (Phase 2)
- New `video_evaluator/road_boundaries.py` (`RoadBoundaryTracker`): per-row asphalt edges above the bike, shared-vanishing-point model `x = x_vp + s*t + k*t^2`, robust fit, temporal smoothing, per-edge cut at vehicles / the bike. The old pipeline clamped the vanishing point to y in [0.48, 0.56], which is wrong for this camera (true VP is near y = 0.05-0.15).
- Telemetry/HUD carry the new `road_edges` (`left`/`right` -> `ys`,`xs`); the legacy `bound_*` arrays are untouched (carpet/ticks still use them).
- UI: toggle `Road Boundaries (Cyan)`; new `Isolate Road Boundaries (Cyan)` button.

## 3. Verification on `test2pothole.mp4` (1280x720, 349 frames, 60 fps)
| Check | Result |
| :--- | :--- |
| Raw model confidence, full video, every 2nd frame | max 0.81; never >= 0.90 |
| Detections at raw >= 0.45 (potholes-only layer) | 25 of 175 sampled frames (156-170, 208-212, 232-244, 264-278); every box visually checked: all on real potholes/puddles, tracked as they approach; **0 boxes** on bus, trees, shadows, bike, road edges |
| Not detected at any confidence | extra puddles at the left edge of frames 232-242 and 266-272 |
| Boundaries (boundaries-only layer) | valid from frame 4; VP x 0.47-0.55, y 0.05-0.16; edges never cross, widen downward, converge exactly at VP; visible edge jitter <= 1.9% of width per 2 frames |
| Synthetic ground-truth roads (straight / right / left bend), 8 frames | VP error <= 0.012, edge error <= 1.4% of width; curvature recovered but damped (0.32 vs 0.45 after 8 frames) |

## 4. Tests / not run here
- Run here (Linux, torch CPU): T-UX-6, T-UX-10, T-UX-11, T-UX-11b pass. Pipeline smoke run with a stubbed Rust lib passes (threshold 0.75, `road_edges` on every frame).
- **Not run**: T-UX-7/8/9 (need `libpothole_core.dylib`, macOS) and the live web UI. T-UX-7/9 were edited for the 0.45 gate.
- T-UX-10 / T-UX-11b read the real video from `$POTHOLE_TEST_VIDEO` (default `~/Downloads/test2pothole.mp4`) and skip if absent.

## 5. Known limits
- The 0.45 gate is low for a detector: it relies on this model producing no false boxes on this footage (verified on this video only). Raise `DEFAULT_CONF_THRESHOLD` (and `YOLO_CONF` in `app.js`) if other footage shows false boxes.
- Boundaries assume the ego bike occupies the lower part of the frame (`SCAN_Y_MAX = 0.44`) and detect vehicles only as "pixels that are not road, terrain or paint".
- The first ~3 frames of a video draw no boundaries (tracker warm-up).
- The green carpet / distance ticks / horizon still use the older geometry model and its VP clamp.

## 6. Fix: far potholes were dropped by the road gate (`fix/pothole-corridor-horizon-gate`)
- **Symptom**: a clear mid-distance pothole (frame 79 of `test2pothole.mp4`) had no box; HUD showed `VP:(0.50,0.50)`, `Road Hazards:0`.
- **Cause**: after the model and Rust NMS, `server.py` dropped every box where Rust `is_point_in_drivable_corridor` returned false. That check needs `y_horizon <= y_bottom <= 0.75`, and `y_horizon` is the Phase 8 horizon clamped to `[0.48, 0.56]` (measured true horizon is ~0.05). Any pothole whose bottom edge is above mid-frame (far) or below 75% of the frame (very close) was removed.
- **Change**: the drop decision now uses the measured road edges (`road_boundaries.point_on_measured_road`); the Rust corridor result is only used for `in_path`. HUD header shows `model N > rust N` counts so a drop stage is visible.
- **Tests**: `user_test_experience/tests/test_pothole_road_gate.py` (8 passed). Not run here: the model, `libpothole_core.dylib`, T-UX-7/8/9 (need macOS) -- re-upload `test2pothole.mp4` after restarting the evaluator.

## 7. Fix: Horizon/Distance Calibration & Async Cloud Evaluator (`fix/road-horizon-and-distance-calibration`)
- **Distance & Horizon Fix**: Wired `RoadBoundaryTracker` (`road["vp"]`) and new `detect_ego_road_bottom` / `estimate_road_distance_meters` into `process_video_through_rust_pipeline`, `extract_onroad_obstacles_for_carpet`, `compute_phase8_green_carpet_polygon`, and `compute_perspective_distance_ticks`. Eliminates mid-frame horizon clamping (`30.0m` pothole distances), ticks/carpet over bike handlebars, and false `2.0m`/`14.9m` ego-dash obstacle alerts.
- **Async Job Queue & Proxy Timeout Fix**: Added `POST /api/process-video-async`, `POST /api/process-fixture-async`, and `GET /api/job/<job_id>` in `server.py` with live progress polling + safe non-JSON HTTP error parsing in `app.js` (resolving `503 Service Unavailable` / `Unexpected token 'S'... is not valid JSON` behind Cloud Shell / Cloud Run proxies). Updated `Dockerfile` and `DEPLOY_GCP.md` with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `--memory 4Gi`, `--timeout 900`, and `--no-cpu-throttling`.
- **Tests**: `pytest user_test_experience/tests/test_async_eval_jobs.py user_test_experience/tests/test_distance_calibration.py user_test_experience/tests/test_pothole_road_gate.py -v` -> **13 passed** (`0` failed); `cargo test --manifest-path pothole_core/Cargo.toml` -> **18 unit + 1 bench passed** (`0` failed).
