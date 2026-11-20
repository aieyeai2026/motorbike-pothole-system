# STATUS — Phase 6: Automotive AR HUD, Curved Road Boundaries & Trajectory Collision Corridor
Date: 2026-11-20

## 1. Summary of Changes
1. **Road-Level Horizon Calibration (`calibrate_road_horizon_norm`)**:
   - Calibrates $y_{\text{horizon}}$ strictly to the physical road-level vanishing horizon ($50\% - 58\%$ down the frame) where continuous low-saturation gray asphalt begins, ignoring sky, trees, and building roofs above it.
2. **Dynamic 2nd-Degree Polynomial Curved Road Boundaries (`QuadraticCurve`, `fit_curved_road_boundaries`)**:
   - Replaced fixed straight trapezoid lines with dynamic quadratic polynomial curves $x_{\text{left}}(y) = a_1 y^2 + b_1 y + c_1$ and $x_{\text{right}}(y) = a_2 y^2 + b_2 y + c_2$ fitted via sliding horizontal asphalt scanlines.
3. **Strict Drivable Road Surface Filtering (`is_point_in_drivable_corridor`, `pothole_core_classify_hazard_corridor`)**:
   - Hard-filters all candidate bounding boxes against the curved drivable road polygon in compiled Rust C-ABI plus surrounding gray-asphalt ring verification, eliminating off-road false detections on parked cars, fences, walls, and trees.
4. **Dynamic Backup-Camera Trajectory Guidance (`compute_trajectory_corridor`, `inverse_perspective_y_for_distance`)**:
   - Projects curved left/right tire-track guidelines (~0.9m motorcycle corridor) bending dynamically with road steering curvature, with horizontal distance rungs at **5m, 10m, and 15m**.
5. **Collision Path Intersect Warning (`intersects_trajectory_corridor`)**:
   - Out-of-path road defects render as unobtrusive 4px cyan dots; potholes `< 10m` directly intersecting the motorcycle's trajectory path highlight the track in crimson (`#FF1744`), render an impact target reticle, and trigger `⚠️ POTHOLE IN PATH: {distance}m`.

## 2. Test Counts (Quoted)
- `cargo test --manifest-path pothole_core/Cargo.toml`:
  `"test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"` (+1 benchmark test)
- `cd mobile_app && flutter test`:
  `"00:00 +12: All tests passed!"`
- `pytest user_test_experience/tests/test_user_experience.py pothole-sync-backend/`:
  `"17 passed, 1 warning in 0.99s"`
