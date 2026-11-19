# STATUS — Phase 3: Python FastAPI Telemetry & 3-Meter Spatial Clustering Backend (`pothole-sync-backend`)
Date: 2026-11-19

## 1. PRs in Merge Order
1. PR #5 (`feat/task-3-1-fastapi-schemas-health` -> `main`, merge commit `9972668`): Backend project scaffolding, Pydantic v2 schemas (`SyncBatchRequest`, `PotholeDetectionItem` with <=100 KB thumbnail validation, `SyncResponse`), and `GET /health` endpoint (`T-3.1.1`..`T-3.1.3`). Also integrates Antigravity PR & git workflow guidelines (`AGENTS.md`, `.agents/rules/git-workflow.md`) and CI suite (`.github/workflows/ci.yml`).
2. PR #6 (`feat/user-test-experience` -> `main`, merge commit `d349143`): User test experience suite (`user_test_experience/`), real physical device testing manual (`MANUAL_DEVICE_TESTING.md`), interactive tactical Web Console (`/console`), CLI motorbike ride simulator (`simulate_ride.py`), backend launcher with LAN IP auto-detection (`start_test_server.py`), and mobile app runtime backend URL settings dialog.
3. PR #7 (`feat/task-3-2-spatial-clustering-db` -> `main`): SQLAlchemy persistent database models (`pothole_events`, `clustered_hazards`), 3.0-meter Haversine spatial clustering engine, event idempotency verification (`T-3.2.1`..`T-3.2.3`), dynamic road border & lane centering in Rust (`pothole_core::geometry::compute_lane_centering`), and simulator ego-positioning.

## 2. Test Counts (Quoted)
- `cd pothole_core && cargo test`:
  `"test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"`
- `cd mobile_app && flutter test`:
  `"00:00 +5: All tests passed!"`
- `cd pothole-sync-backend && pytest`:
  `"9 passed, 1 warning in 0.44s"`
- `pytest user_test_experience/tests/test_user_experience.py`:
  `"4 passed in 0.06s"`

## 3. CI Result
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) passed on PR #5 & PR #6 (`✓ 2 successful`).

## 4. What Is Not Proven Yet
- Real-world motorbike engine vibration tolerance on camera OIS without active dampener mount.
- Real-time video frame rate on physical mid-range Android phone under 35°C outdoor direct sunlight.

## 5. To Do By Hand
- Execute physical road test scenarios detailed in `user_test_experience/MANUAL_DEVICE_TESTING.md` on motorbike mount.

## 6. Not Built Yet
- Task 3.3: Async Gemini 2.5 Flash false-positive triage worker (`T-3.3.3`) and production PostgreSQL migration.

## 7. Open Decisions
- Standardized 3.0m clustering radius across all components; lane centering dynamically calculates lateral offset against detected boundaries ($x_{left}$, $x_{right}$) with a $\pm 0.25\text{m}$ deadband for center classification.
