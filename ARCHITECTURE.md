# Architecture: Motorbike Edge Vision & Telemetry System

## 1. System Overview
The system detects, estimates distance to, and clusters road hazards (potholes) in real time for motorbike riders.
The pipeline consists of three tiers:
1. **Edge Vision Core (`pothole_core`)**: Rust library compiled to C-ABI shared library for zero-copy preprocessing, YOLOv8 tensor parsing, NMS, IMU severity scoring, and distance/lane geometry.
2. **Mobile Client (`mobile_app`)**: Flutter app hosting camera feed, rendering a tactical HUD with <8m critical alerts, and maintaining an offline SQLite queue with 3.0m spatial deduplication and batch syncing.
3. **Telemetry & Spatial Clustering Backend (`pothole-sync-backend`)**: Python FastAPI service ingesting telemetry batches, performing 3.0m Haversine spatial clustering into persistent hazard entities, serving nearby hazards, and performing async Gemini 2.5 Flash thumbnail false-positive triage.

```
+--------------------------------------------------------------------+
|                         Motorbike Mount                            |
|                                                                    |
|  [Camera + IMU]                                                    |
|         │                                                          |
|         ▼                                                          |
|  [Flutter Mobile App (mobile_app)]                                 |
|         │                                                          |
|         ├─► Dart FFI ──► [pothole_core (Rust C-ABI)]               |
|         │                  ├─ Sky ROI Crop & Float Norm            |
|         │                  ├─ YOLOv8 Tensor Parser [1, 5, 8400]    |
|         │                  ├─ NMS (IoU >= 0.45)                    |
|         │                  ├─ Distance Math & Lane Classification  |
|         │                  └─ IMU Z-Peak Severity Fusion           |
|         │                                                          |
|         ├─► Tactical HUD (Canvas overlay, <8m Critical Alert)      |
|         │                                                          |
|         └─► Offline SQLite Queue (LocalHazardQueue)                |
|               ├─ 3.0m Haversine Edge Deduplication                 |
|               └─ Batch Flush (≤ 50 records / request)              |
+--------------------------------------------------------------------+
                                  │
                                  │ HTTP POST /api/v1/telemetry/sync
                                  ▼
+--------------------------------------------------------------------+
|             FastAPI Backend (pothole-sync-backend)                 |
|                                                                    |
|  ├─ Pydantic v2 Validation (SyncBatchRequest, ≤100 KB thumbnail)   |
|  ├─ Event UUID Idempotency Check                                   |
|  ├─ 3.0m Haversine Spatial Clustering Engine                       |
|  ├─ Database (SQLAlchemy: pothole_events, clustered_hazards)       |
|  ├─ Spatial Hazard Query: GET /api/v1/hazards/nearby               |
|  └─ Async Gemini 2.5 Flash False-Positive Thumbnail Triage         |
+--------------------------------------------------------------------+
```

## 2. Component Specifications

### 2.1 Rust Edge Vision & Geometry Core (`pothole_core`)
- **Distance Estimation (`geometry.rs`)**: Inverse perspective projection from normalized image bottom $y \in [y_{horizon}, 1.0]$ to distance in meters: $d = k / (y - y_0)$. Calibrated monotonically between 2.0m and 35.0m.
- **Lane Classification (`geometry.rs`)**: X-center bounding box normalized into thirds:
  - $x < 0.33 \implies$ `LEFT` (code 0)
  - $0.33 \le x \le 0.66 \implies$ `CENTER` (code 1)
  - $x > 0.66 \implies$ `RIGHT` (code 2)
- **Zero-Copy ROI & Normalization (`preprocess.rs`)**: Strips top sky region (above horizon, typically top 40%) and normalizes uint8 RGB pixels to $[0.0, 1.0]$ floats.
- **Severity Scoring (`preprocess.rs`)**: Combines detection confidence, bounding box area, and optional vertical IMU accelerometer $z$-axis spike ($g$-forces). Bounded between $1.0$ and $10.0$.
- **Transposed YOLO Tensor Parser (`inference.rs`)**: Parses raw YOLOv8 output tensor formatted as $[1, 5, N]$ (channels: $cx, cy, w, h, conf$), applies confidence thresholding, converts to bounding boxes, and executes Non-Maximum Suppression (IoU $\ge 0.45$).
- **C-ABI FFI (`ffi.rs`)**: Exports `pothole_core_run_postprocess` accepting raw float pointers and returning a contiguous slice of `CDetectedPothole` structs.

### 2.2 Mobile Client & Tactical HUD (`mobile_app`)
- **HUD Interface (`main.dart`)**: Renders custom painted bounding boxes, distance tags, lane indicators, and an immediate red banner if any detected hazard is under 8 meters directly ahead.
- **Dart FFI Bridge (`rust_bridge.dart`)**: Dynamic library binding bridging Flutter isolates with `pothole_core`.
- **Local SQLite Queue (`local_db.dart`)**: Maintains `pending_potholes` SQLite table. Applies a local 3.0m Haversine deduplication filter before inserting new records. Slices at most 50 records per batch to flush to backend telemetry endpoint, deleting successfully acknowledged UUIDs while retaining records on network errors.

### 2.3 Telemetry & Clustering Backend (`pothole-sync-backend`)
- **Pydantic Validation**: `SyncBatchRequest` validates `device_id` and list of `PotholeDetectionItem` records. Enforces decoded thumbnail size $\le 100$ KB.
- **Spatial Deduplication & Clustering**: `CLUSTER_RADIUS_METERS = 3.0`.
  - When a detection arrives, query existing clustered hazards within 3.0 meters using Haversine formula.
  - If match found: update cluster aggregate stats (detection count, running average latitude/longitude, maximum severity, last observed timestamp), link `pothole_events` foreign key.
  - If no match found: create a new `clustered_hazards` entry centered at the detection coordinate.
- **Idempotency**: Detections with previously ingested `event_uuid` are safely skipped without duplicate cluster increments.
- **AI Triage (`google-genai`)**: For detections with thumbnails, an async task sends the thumbnail to Gemini 2.5 Flash to verify whether the image represents an actual road defect vs a benign artifact (shadow, manhole cover, puddle).
