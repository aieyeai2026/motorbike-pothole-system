# STATUS — Phase 8: AR "Green Carpet" Optimal Path & Tilting Horizon HUD
Date: 2026-11-21

## 1. Summary of Changes
1. **Dynamic Vanishing Point & Tilting Green Horizon (`compute_vanishing_point_and_bank`, `compute_phase8_horizon_and_boundaries`)**:
   - Computes road vanishing point $(x_{\text{vp}}, y_{\text{vp}})$ from left and right asphalt edge tangent intersections ($y_{\text{vp}} \in [0.48, 0.58]$) with temporal smoothing ($0.85 \cdot \vec{p}_{vp}^{(t-1)} + 0.15 \cdot \vec{p}_{vp}^{\text{meas}}$).
   - Estimates road banking tilt $\theta_{\text{bank}} = \arctan((m_{\text{left}} + m_{\text{right}})/2)$ and renders the `#00E676` green horizon line centered at $(x_{\text{vp}}, y_{\text{vp}})$ rotated by $\theta_{\text{bank}}$.
2. **Cyan Curved Road Boundaries (`eval_cyan_road_boundary`)**:
   - Evaluates smooth `#00E5FF` road edges with $1.3$ perspective exponent and $\kappa_{\text{road}} \sin(\pi t)$ curvature, guaranteed to converge cleanly at $(x_{\text{vp}}, y_{\text{vp}})$ without crossing.
3. **Solid Green Optimal Trajectory Carpet (`compute_carpet_bounds`, `pothole_core_eval_green_carpet_row`, `compute_phase8_green_carpet_polygon`)**:
   - Tapered mesh ($W_{\text{near}} = 0.18$ at $y_{\text{bot}} = 0.88$ to $W_{\text{far}} = 0.02$ at $y_{\text{target}}$, exponent $1.4$), rendered at 38% opacity (`cv2.addWeighted`) with 2px neon green borders.
   - Dynamically weaves toward open drivable asphalt gaps around vehicles, scooters, pedestrians, and potholes, truncating cleanly at $y_{\text{target}} = \max(y_{\text{obs}}, y_{\text{vp}} + 0.03)$ without drawing over obstacles.
   - Transitions to crimson red (`#FF1744`) with `BRAKE / COLLISION HAZARD` banner when an unavoidable obstacle is $< 8\text{m}$ ahead in path.
4. **9-Frame Verification (`benchmarks/fixtures/phase8_verification/`)**:
   - Rendered and verified `frame_01.jpg`..`frame_09.jpg` across straight roads, bends, oncoming trucks, scooters, and lane splits.

## 2. Test Counts (Quoted)
- `cargo test --manifest-path pothole_core/Cargo.toml`:
  `"test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"` (+1 benchmark test)
- `cd mobile_app && flutter test`:
  `"00:00 +13: All tests passed!"`
- `pytest user_test_experience/tests/test_user_experience.py pothole-sync-backend/`:
  `"18 passed, 1 warning in 1.22s"`
