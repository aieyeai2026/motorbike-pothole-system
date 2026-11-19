# Test Plan: Motorbike Pothole Detection & Telemetry System

## Test Matrix by Phase

### Phase 1: Rust Edge Vision & Geometry Core (`pothole_core`)
- `T-1.1.1`: `estimate_distance_meters` returns monotonic, bounded distances ($2.0\text{m} \le d \le 35.0\text{m}$) across vertical image coordinates.
- `T-1.1.2`: `classify_lane` accurately partitions normalized horizontal span into `LEFT`, `CENTER`, and `RIGHT`.
- `T-1.1.3`: `compute_iou` handles identical, disjoint, and intersecting bounding boxes.
- `T-1.1.4`: `non_max_suppression` filters boxes below confidence threshold and merges overlapping boxes with $\text{IoU} \ge 0.45$.
- `T-1.2.1`: `crop_and_normalize_roi` strips sky pixels above horizon and normalizes RGB `u8` bytes to $[0.0, 1.0]$.
- `T-1.2.2`: `compute_severity_score` stays bounded $[1.0, 10.0]$ and scales monotonically with IMU $z$-axis acceleration spikes.
- `T-1.3.1`: `run_yolo_postprocess` decodes transposed $[1, 5, N]$ YOLOv8 output tensor and executes NMS.

### Phase 2: Flutter HUD, Rust FFI & Offline Queue (`mobile_app`)
- `T-2.1.1`: `pothole_core_run_postprocess` C-ABI FFI writes detections into caller-provided out pointer buffer without memory leaks.
- `T-2.1.2`: `RustPotholeBridge` Dart FFI bindings decode YOLOv8 tensor and map detections to Dart `PotholeHazard` objects.
- `T-2.2.1`: `enqueueIfNotDuplicate` inserts new detection and rejects duplicate detections within 3.0 meters.
- `T-2.2.2`: `flushBatchToBackend` slices at most 50 pending records per HTTP POST sync request.
- `T-2.2.3`: `flushBatchToBackend` retains unsynced records in SQLite table upon backend failure or network timeout.
- `T-2.3.1`: Tactical HUD renders critical alert banner when hazard distance is $< 8.0$ meters.

### Phase 3: Python FastAPI Telemetry & 3-Meter Spatial Clustering Backend (`pothole-sync-backend`)
- `T-3.1.1`: `SyncBatchRequest` validates valid payloads with single and multiple detections, generating correct pydantic model instances.
- `T-3.1.2`: `PotholeDetectionItem` rejects thumbnail payloads where base64 decoded byte length exceeds 100 KB ($102,400$ bytes).
- `T-3.1.3`: `GET /health` returns status `200 OK` with JSON `{"status": "healthy"}`.
- `T-3.2.1`: Spatial clustering creates new `clustered_hazards` entry for initial detection and updates running centroid and detection count for detections within 3.0 meters.
- `T-3.2.2`: Spatial clustering treats detections beyond 3.0 meters as distinct independent hazard clusters.
- `T-3.2.3`: Ingestion enforces `event_uuid` idempotency: resubmitting identical UUID returns acknowledged status without duplicating event or cluster stats.
- `T-3.3.1`: `POST /api/v1/telemetry/sync` ingests batch, commits events and clusters to database, and returns list of synced UUIDs.
- `T-3.3.2`: `GET /api/v1/hazards/nearby` returns only clusters within specified query radius of given coordinate.
- `T-3.3.3`: Async Gemini triage worker invokes `google-genai` model `gemini-2.5-flash` with thumbnail, classifying road defects vs false positives.
