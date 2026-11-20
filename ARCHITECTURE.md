# Architecture: Motorbike Edge Vision & Telemetry System

## 1. System Overview

The Motorbike Pothole System detects, estimates distance to, and clusters road hazards (potholes, craters, surface fractures, and roadway obstacles) in real time for motorbike riders. Designed for rugged, high-vibration motorcycle environments, the architecture combines on-device computer vision, physical inertial sensor fusion, an offline-first SQLite persistence layer, and a cloud-based spatial clustering backend.

The system spans four integrated tiers:
1. **Edge Vision Core (`pothole_core`)**: Zero-copy Rust library compiled to a C-ABI static/dynamic library for ROI preprocessing, YOLOv8 tensor parsing, NMS, IMU severity scoring, perspective distance math, and dynamic lane centering.
2. **Mobile Client & Tactical HUD (`mobile_app`)**: Flutter application with native camera streaming, an Apple iOS / Android sensory telemetry engine, a 30 FPS tactical cockpit HUD with `< 8m` crimson collision alerts, and an offline SQLite queue with 3.0m Haversine deduplication.
3. **iPhone 12 Sensory Telemetry Engine**: Hardware-level CoreMotion (3-axis accelerometer vertical shock $G_z$ and gyroscope roll velocity) and CoreLocation (GNSS latitude, longitude, and speedometer) streams fused into detection severity and lane tracking.
4. **Telemetry & Spatial Clustering Backend (`pothole-sync-backend`)**: Python FastAPI service ingesting telemetry batches, executing 3.0-meter Haversine spatial clustering, serving nearby hazard queries, and running asynchronous Gemini 2.5 Flash thumbnail false-positive verification.

```
+-----------------------------------------------------------------------------------------+
|                                MOTORBIKE COCKPIT MOUNT                                  |
|                                                                                         |
|  [iPhone 12 Hardware Sensors]                                                           |
|    ├─ AVFoundation Camera ──► YUV420 Frames (30/60 FPS)                                  |
|    ├─ CoreMotion IMU      ──► Accelerometer Z-Shock (G_z) + Gyroscope Roll Rate (deg/s) |
|    └─ CoreLocation GNSS   ──► Lat, Lon Coordinates + Speedometer (km/h) + Heading (deg) |
|               │                                                                         |
|               ▼                                                                         |
|  [Flutter Mobile App (mobile_app)]                                                      |
|    │                                                                                    |
|    ├─► Dart FFI ──► [pothole_core (Rust C-ABI)]                                         |
|    │     ├─ Zero-Copy Sky ROI Crop & Float Normalization                                |
|    │     ├─ YOLOv8 Tensor Parser [1, 5, 8400] / Multi-Class [1, 4+C, N]                 |
|    │     ├─ Non-Maximum Suppression (IoU >= 0.45)                                        |
|    │     ├─ Inverse Perspective Distance Projection: d = k / (y - y_0)                  |
|    │     ├─ Dynamic Lane Boundary Corridor Classification (LEFT, CENTER, RIGHT)         |
|    │     ├─ Dynamic Lateral Centering against Curbs: Δx = (x_ego - x_center) * W_road   |
|    │     └─ Physical IMU Z-Peak Severity Fusion (G-force impact scaling: 1.0 - 10.0)    |
|    │                                                                                    |
|    ├─► Tactical Motorcycle HUD Overlay (Canvas / WebRTC / 30 FPS Custom Painter):       |
|    │     ├─ 3D Perspective Road Curbs & Animated Travel Corridors                       |
|    │     ├─ Dynamic Pothole Bounding Boxes with Distance Tags (e.g. "4.8m [CENTER]")    |
|    │     ├─ Immediate Crimson Alert Banner when Hazard Distance < 8.0 Meters            |
|    │     ├─ Live Sensory Telemetry Bar: GPS, Speedometer, IMU G-Force Gauge, Heading    |
|    │     └─ Ego Motorcycle Handlebar Crosshair Reticle & Lateral Clearance Badges       |
|    │                                                                                    |
|    └─► Local Offline SQLite Queue (LocalHazardQueue):                                   |
|          ├─ 3.0m Edge Haversine Deduplication Filter (retains single closest event)     |
|          ├─ Auto-captures thumbnail image (≤ 100 KB base64 JPEG)                        |
|          └─ Batch Flush Engine (slices ≤ 50 records per HTTP POST sync request)         |
+-----------------------------------------------------------------------------------------+
                                           │
                                           │ HTTP POST /api/v1/telemetry/sync
                                           │ (Dual Wi-Fi LAN / Cellular / USB adb)
                                           ▼
+-----------------------------------------------------------------------------------------+
|                    TELEMETRY & SPATIAL CLUSTERING BACKEND (FastAPI)                     |
|                                                                                         |
|  ├─ Pydantic v2 Contract Validation (SyncBatchRequest, ≤ 100 KB image thumbnail check)  |
|  ├─ Event UUID Idempotency Engine (rejects replayed detections with zero duplicate rows)|
|  ├─ 3.0-Meter Haversine Spatial Clustering Engine (CLUSTER_RADIUS_METERS = 3.0):        |
|  │     ├─ Queries existing clusters within 3.0m of incoming lat/lon                     |
|  │     ├─ Matches existing: updates running centroid, detection count, max severity     |
|  │     └─ Unmatched: seeds a new clustered_hazards record centered at coordinates       |
|  ├─ Relational Persistence (SQLAlchemy: pothole_events, clustered_hazards)              |
|  ├─ Spatial Hazard Query API: GET /api/v1/hazards/nearby?lat=...&lon=...&radius=100m   |
|  └─ Async Gemini 2.5 Flash Thumbnail Triage Worker (google-genai):                     |
|        └─ Classifies thumbnail as TRUE_HAZARD vs FALSE_POSITIVE (shadow, manhole, etc.)|
+-----------------------------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 Rust Edge Vision & Geometry Core (`pothole_core`)

Compiled as a zero-dependency C-ABI shared/static library (`libpothole_core.dylib`, `libpothole_core.so`, `libpothole_core.a`):

- **Inverse Perspective Distance Projection (`geometry.rs`)**:
  Maps normalized bounding box base coordinate $y \in [y_{\text{horizon}}, 1.0]$ to physical ground distance in meters:
  $$d = \frac{k}{y - y_0}$$
  Where $y_0 \approx 0.40$ (calibrated horizon line) and $k \approx 4.0$ (focal scaling constant). Monotonically bounded between $2.0\text{m}$ (immediate front tire) and $35.0\text{m}$ (far sight distance).
- **Lane Classification (`geometry.rs`)**:
  Calculates normalized horizontal center $x_c = (x_{\min} + x_{\max}) / 2$:
  - $x_c < 0.33 \implies$ `LEFT` (code 0)
  - $0.33 \le x_c \le 0.66 \implies$ `CENTER` (code 1)
  - $x_c > 0.66 \implies$ `RIGHT` (code 2)
- **Dynamic Lane Centering (`geometry.rs`)**:
  Evaluates lateral motorbike offset from detected road boundaries ($x_{\text{left}}, x_{\text{right}}$):
  $$x_{\text{center}} = \frac{x_{\text{left}} + x_{\text{right}}}{2}, \quad \Delta x = (x_{\text{ego}} - x_{\text{center}}) \times W_{\text{road}}$$
  Classifies rider position as `CENTERED` if $|\Delta x| \le 0.25\text{m}$, or indicates directional drift (`DRIFT LEFT` / `DRIFT RIGHT`).
- **Zero-Copy Sky ROI Crop & Float Normalization (`preprocess.rs`)**:
  Strips the top sky region ($y < y_{\text{horizon}}$) to eliminate irrelevant lighting artifacts (clouds, overhead foliage) and normalizes uint8 RGB pixels to $[0.0, 1.0]$ float tensors without heap reallocations.
- **IMU Z-Peak Severity Fusion (`preprocess.rs`)**:
  Merges visual bounding box area and detection confidence with vertical accelerometer shock ($G_z$):
  $$\text{Severity} = \text{clamp}\left(1.0 + 4.5 \times \text{conf} + 2.5 \times \text{area}_{\text{norm}} + 2.0 \times \max(0, G_z - 1.0), \; 1.0, \; 10.0\right)$$
- **Transposed YOLO Tensor Parser (`inference.rs`) & NMS (`nms.rs`)**:
  Decodes raw YOLOv8 output tensor formatted as $[1, 5, N]$ (channels: $cx, cy, w, h, \text{conf}$), applies confidence thresholding ($\ge 0.50$), and executes Non-Maximum Suppression with IoU threshold $\ge 0.45$.
- **C-ABI FFI Boundary (`ffi.rs`)**:
  Exports `pothole_core_run_postprocess` accepting raw float pointers and writing contiguous `CDetectedPothole` structs for zero-copy FFI interop with Dart/Flutter and Python.

---

### 2.2 Mobile Client & Tactical HUD (`mobile_app`)

Written in Flutter with native iOS and Android platform channels:

- **Tactical Motorcycle HUD (`main.dart`)**:
  - Immersive landscape cockpit overlay designed for handlebar mounts.
  - Draws 3D perspective road corridors, speed-animated lane dashes, depth distance gauge lines (10m, 20m, 30m), and the motorcycle steering reticle.
  - Renders dynamic bounding boxes with high-contrast distance and lane tags (`"7.3m [CENTER]"`).
  - Triggers an immediate flashing crimson banner and audible chime when any hazard is $< 8.0\text{m}$ directly ahead.
- **Offline SQLite Queue (`local_db.dart`)**:
  - Maintains the `pending_potholes` SQLite table.
  - Applies a 3.0-meter Haversine deduplication filter on incoming detections before insertion, preventing duplicate storage when tracking the same pothole over consecutive frames.
  - Slices at most 50 records per batch to flush to the backend (`POST /api/v1/telemetry/sync`), deleting acknowledged records and retaining unacknowledged entries on network loss.

---

### 2.3 iPhone 12 Hardware Sensory Architecture & Fusion Engine

The mobile client interfaces directly with Apple's hardware sensors:

| Sensor Hardware | Apple Framework | Data Stream | System Utilization |
|---|---|---|---|
| **3-Axis Accelerometer** | **CoreMotion** | Vertical Linear Shock ($a_z \to G_z$) | Measures suspension impact force when rolling over a defect. Fuses transient peak $G$-forces ($1.0\text{G} \to 2.5\text{G}+$) directly into the pothole severity score. |
| **3-Axis Gyroscope** | **CoreMotion** | Roll Angular Velocity ($\omega_x$ deg/s) | Tracks motorcycle banking/lean angle during turns. Allows dynamic horizon tilting and lane corridor boundary compensation. |
| **GNSS Receiver** | **CoreLocation** | Latitude, Longitude, Ground Speed | Tags detections with accurate geographical coordinates for 3.0m spatial deduplication and backend cluster mapping. Live GPS speed calculates Time-to-Collision ($TTC = \text{dist} / \text{speed}$). |
| **Magnetometer** | **CoreLocation** | Compass Heading ($0^\circ - 360^\circ$) | Records directional heading of travel along the roadway corridor. |
| **12MP Wide Camera** | **AVFoundation** | 30/60 FPS YUV420 Stream | Feeds continuous optical frames to the zero-copy ROI preprocessor and neural network inference engine. |

---

### 2.4 Telemetry & Clustering Backend (`pothole-sync-backend`)

Built with Python 3.12, FastAPI, and SQLAlchemy:

- **Pydantic Validation**:
  `SyncBatchRequest` validates `device_id` and detection items, strictly enforcing that decoded thumbnail byte sizes do not exceed $100\text{ KB}$.
- **Spatial Deduplication & Clustering Engine (`clustering.py`)**:
  - Standardized cluster radius: `CLUSTER_RADIUS_METERS = 3.0`.
  - On detection ingestion, calculates Haversine distances against existing clustered hazards:
    $$d = 2R \arcsin \left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
  - If match within 3.0m exists: updates cluster running average coordinates, increment counter, maximum severity, and latest observation timestamp.
  - If no match exists: seeds a new cluster entity centered at the detection coordinate.
- **Idempotency**: Detections with existing `event_uuid` values are safely acknowledged without creating redundant database records.
- **AI Triage (`google-genai`)**: Background worker passes thumbnails to Gemini 2.5 Flash to verify true road damage vs benign artifacts (tree shadows, manhole lids, painted crosswalks).

---

## 3. Phase 4 Architecture: Multi-Class Hazards, Road Edges & Biker Position

Phase 4 extends the detection core from single-class potholes to multi-class road obstacle detection and dynamic motorcycle positioning.

### 3.1 Multi-Class Tensor Parser & Per-Class NMS
- **Tensor Format**: Extends tensor decoding from $[1, 5, N]$ to $[1, 4+C, N]$ where $C \ge 4$ classes:
  1. `0: pothole` (road surface crater / structural void)
  2. `1: speed_bump` (raised asphalt / painted rumble strip)
  3. `2: road_debris` (fallen cargo, tire tread, construction rubble)
  4. `3: vehicle` (cars, trucks, auto-rickshaws, motorbikes ahead)
- **Class-Aware NMS**: Suppresses overlaps independently per class so an ahead vehicle does not suppress a road pothole directly in front of it.
- **Database Partitioning**: Vehicles and moving obstacles are alerted on the HUD but never persisted to the cloud pothole database (`hazard_type == 'pothole'` only).

### 3.2 Dynamic Road Boundary & Lateral Clearance Guidance
- **Road Boundary Fitting**: Extracts left curb ($x_{\text{left}}$) and right curb ($x_{\text{right}}$) lines converging to the horizon.
- **Biker Lateral Positioning**: Calculates front wheel ground contact offset $\Delta x$ relative to road center:
  $$\Delta x = (x_{\text{ego}} - x_{\text{center}}) \times W_{\text{road}}$$
- **HUD Reticle**: Displays lateral road clearance gauges (`L: 1.8m | CENTERED | R: 1.8m`) with real-time drift warnings (`DRIFT LEFT`, `DRIFT RIGHT`).

---

## 4. Phase 5 Architecture: On-Device Real Model Inference & Field Validation

Phase 5 replaces simulated test tensors with **real-time on-device neural network inference on live camera frames**.

### 4.1 On-Device Model Architecture & Hardware Acceleration
```
Raw Camera Frame (1280x720 YUV420) 
   │
   ▼ [Sky ROI Crop: Top 40% removed in Rust]
Road ROI (1280x432)
   │
   ▼ [Bilinear Resize & Planar Transpose]
Model Input Tensor: [1, 3, 640, 640] FP16 / INT8
   │
   ▼ [Inference on Apple Neural Engine (ANE) via CoreML]
Raw Output Tensor: [1, 4+C, 8400]
   │
   ▼ [Rust C-ABI Post-Processing: pothole_core_run_postprocess]
CDetectedPothole Array (NMS, Distance, Lane, IMU Fusion)
```
- **Apple CoreML Model (`.mlpackage` / `.mlmodelc`)**: Quantized YOLOv8-nano fine-tuned on road defect datasets, compiled for the iPhone 12 Apple Neural Engine (16-core ANE).
- **Android / TFLite (`.tflite`)**: Quantized INT8 model executing via GPU delegates.
- **Latency Budgets**:
  - Preprocessing (Rust): $\le 2\text{ ms}$
  - ANE Inference: $\le 15\text{ ms}$
  - Postprocessing (Rust NMS + Distance): $\le 1\text{ ms}$
  - Total frame-to-HUD alert latency: $\le 34\text{ ms}$ ($\ge 30\text{ FPS}$).

### 4.2 On-Device False-Positive Suppression
Heuristics to suppress false alarms identified in field footage (`docs/benchmarks/VIDEO_ANALYSIS_PHASE5.md`):
1. **Canopy Shadow Rejection**: Tree shadows have high contrast but zero vertical depth gradient and generate no IMU shock.
2. **Paint Marking Rejection**: Crosswalk stripes and road text arrows are rejected via aspect ratio and interior texture flatness.
3. **Paver Block Texture Transition**: Repeated geometric joints are filtered out using spatial variance thresholding.

---

## 5. Phased Roadmap & Milestone Matrix

| Phase | Milestone Name | Key Deliverables | Status |
|---|---|---|---|
| **Phase 1** | **Rust Edge Vision Core** | Distance math, lane classification, NMS, sky crop, IMU severity scoring, YOLOv8 parser. | **COMPLETED** (PRs #1, #2, #3) |
| **Phase 2** | **Flutter App & Offline DB** | Rust C-ABI FFI bridge, Tactical HUD canvas, `< 8m` critical alert, offline SQLite queue, 3.0m deduplication. | **COMPLETED** (PRs #4, #5) |
| **Field UX** | **Device & Sensory Integration** | Web Console test bench, Safari PWA mode, dual HTTPS streaming, **iPhone 12 CoreMotion (IMU) + CoreLocation (GPS/Speed) integration**. | **COMPLETED** (PRs #6 – #12) |
| **Phase 3** | **FastAPI Spatial Backend** | Task 3.1 (Schemas/Health), Task 3.2 (3.0m Haversine Clustering DB), Task 3.3 (Sync endpoints & Gemini triage). | **ACTIVE (~80%)** |
| **Phase 4** | **Multi-Class Hazards & Biker Position** | Multi-class $[1, 4+C, N]$ parser, per-class NMS, road boundary curb detection, dynamic ego lane centering. | **PLANNED NEXT** |
| **Phase 5** | **Real Model & Field Validation** | On-device CoreML / TFLite inference on live frames, ANE hardware acceleration, road testing on motorcycle mount. | **UPCOMING** |
