# STATUS — Phase 5: Hardware Safeguards, False-Positive Suppression & Benchmark Comparison
Date: 2026-11-20

## 1. Summary of Changes
- **Pitch-Compensated Horizon Math (`pothole_core/src/geometry.rs`)**:
  - Added `compensate_horizon_for_pitch(base_horizon_norm, pitch_deg)` to prevent braking dive and acceleration pitch from distorting perspective distance estimates ($d = f(y)$).
- **IMU Engine Vibration Filter & On-Device False-Positive Suppression (`pothole_core/src/preprocess.rs`)**:
  - Added `filter_imu_engine_vibration(samples_g)` to strip high-RPM steady-state handlebar buzz while preserving transient suspension impact peaks.
  - Added `is_likely_false_positive_pothole` to suppress wide crosswalk stripe triggers (`w / h > 5.5`) and low-confidence tree canopy shadows (`conf < 0.60` with flat IMU `<= 1.05G`).
- **Speed-Adaptive & Thermal FPS Controller (`mobile_app/lib/main.dart`)**:
  - Added `computeTargetFps(speedKmh, thermalModerate)` enforcing 0 FPS when stopped (`< 2 km/h`), 10 FPS in city traffic (`< 40 km/h`) or under moderate thermal throttling, and 30 FPS when cruising (`>= 40 km/h`).
- **Part C Benchmark Reports (`docs/benchmarks/TEST_NEW.md`, `docs/benchmarks/TEST_COMPARISON.md`)**:
  - Executed 3 benchmark runs and recorded side-by-side comparison against `TEST_OLD.md`.

## 2. Test Counts (Quoted)
- `cargo test --manifest-path pothole_core/Cargo.toml`:
  `"test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"` (+1 integration benchmark test)
- `cd mobile_app && flutter test`:
  `"00:00 +11: All tests passed!"`
- `pytest pothole-sync-backend/ user_test_experience/tests/`:
  `"14 passed, 1 warning in 0.36s"`
