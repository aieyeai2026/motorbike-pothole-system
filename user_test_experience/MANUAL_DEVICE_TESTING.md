# Motorbike Pothole System — Real Device Testing Manual

This manual provides complete step-by-step instructions for mounting, connecting, calibrating, and field-testing the real-time motorbike pothole detection system with physical mobile devices and motorbikes.

---

## 1. Hardware Architecture & Mounting Setup

### 1.1 Recommended Motorcycle Mount
- **Phone Mount**: QuadLock Motorcycle Handlebar/Fork Stem Mount or RAM Mount with double-socket arm.
- **Vibration Dampener (MANDATORY)**: Modern smartphones feature optical image stabilization (OIS) magnets susceptible to engine harmonic vibration. Always use an active vibration dampening module.
- **Orientation**: Landscape mode centered over the handlebars, aligned perpendicularly to the motorbike's direction of travel.
- **Power**: Standard 12V-to-5V USB motorcycle charger (QC 3.0 or USB-PD, minimum 15W) to sustain 30 FPS camera inference and screen brightness.

```
       [Front Wheel / Road]
                 ▲
                 │ 5m - 35m Detection Range
                 │
      [Handlebars / Cockpit]
   ┌───────────────────────────┐
   │ [Phone (Landscape Mount)] │ ◄── 40% Horizon line calibrated
   │   • Camera facing front   │
   │   • Screen: Tactical HUD  │
   └─────────────┬─────────────┘
                 │ USB Cable (Optional ADB reverse)
                 ▼
          [Power Bank / PC]
```

### 1.2 Horizon Alignment Calibration
The perspective distance equation in `pothole_core/src/geometry.rs` uses:
$$d = \frac{k}{y - y_0}$$
where $y_0 = 0.40$ (the horizon line represents the top 40% of the camera frame, with sky pixels above the horizon cropped out).
- **Physical Check**: When seated on the bike on a level road, ensure the road horizon aligns visually with the top 40% horizontal cyan guide line drawn on the Tactical HUD.

---

## 2. Mobile Device Connection Options

Choose one of the following methods to connect your physical mobile device to the telemetry backend:

### Option A: Local Wi-Fi Network (Workshop / Driveway Testing)
1. Connect both your laptop and phone to the same Wi-Fi network.
2. Start the telemetry backend server:
   ```bash
   python user_test_experience/start_test_server.py
   ```
3. Note your laptop's local LAN IP printed in the banner (e.g. `http://192.168.1.9:8000`).
4. In the Flutter mobile app:
   - Tap the **Settings (gear)** icon in the top header.
   - Enter `http://192.168.1.9:8000` (substitute your machine's actual LAN IP).
   - Tap **Test Ping**. A green snackbar confirms `Health check response: 200 {"status": "healthy"}`.
   - Tap **Save**.

---

### Option B: USB Cable + ADB Reverse Tethering (Zero-Latency, No Wi-Fi Needed)
If Wi-Fi is weak in your garage or driveway:
1. Connect your Android phone to the laptop via USB cable.
2. Enable **USB Debugging** in Android Developer Options.
3. Forward the port from the phone to your laptop:
   ```bash
   adb reverse tcp:8000 tcp:8000
   ```
4. In the Flutter mobile app:
   - Set backend URL to `http://localhost:8000` or `http://127.0.0.1:8000`.
   - The phone routes network requests over the high-speed USB cable directly to your laptop backend.

---

### Option C: Mobile Hotspot (Field Road Test on Motorbike)
When testing on the open road without a fixed Wi-Fi router:
1. Turn on **Personal Hotspot** on your phone.
2. Connect your laptop (placed in backpack or tank bag) to the phone's Wi-Fi hotspot.
3. Run `python user_test_experience/start_test_server.py`.
4. The laptop will detect its IP on the phone's hotspot subnet (typically `192.168.43.x` on Android or `172.20.10.x` on iOS).
5. Point the mobile app to `http://<LAPTOP_HOTSPOT_IP>:8000`.

---

### Option D: Cloud Tunnel / Ngrok (Public HTTPS Endpoint)
To test remotely with your laptop running at home:
1. Start ngrok tunnel:
   ```bash
   ngrok http 8000
   ```
2. Copy the public forwarding HTTPS URL (e.g. `https://abc1234.ngrok-free.app`).
3. Set the backend URL in the mobile app to this HTTPS address.

---

## 3. Step-by-Step Field Test Procedures

### Test Scenario 1: Stationary Bench Calibration
**Objective**: Verify camera bounding box, distance estimation, and lane partition.
1. Place the bike on its center stand on a level driveway.
2. Place a high-contrast target (a $30\text{cm} \times 30\text{cm}$ dark cardboard cutout or chalk pothole) on the ground:
   - At $5.0\text{m}$ ahead (Center lane)
   - At $12.0\text{m}$ ahead (Left lane)
   - At $20.0\text{m}$ ahead (Right lane)
3. Launch the Flutter app (`mobile_app`).
4. **Pass Criteria**:
   - $5.0\text{m}$ marker reports distance between $4.5\text{m} - 5.5\text{m}$ and lane `CENTER`.
   - $12.0\text{m}$ marker reports lane `LEFT`.
   - $20.0\text{m}$ marker reports lane `RIGHT`.

---

### Test Scenario 2: Critical <8m Alert Triggering
**Objective**: Confirm immediate driver warning for imminent impact hazards.
1. Approach a hazard target from $15\text{m}$ at walking/jogging speed ($5-10\text{ km/h}$).
2. Observe the HUD as distance transitions below $8.0\text{m}$.
3. **Pass Criteria**:
   - At $d < 8.0\text{m}$, the bounding box switches from cyan (`#00E5FF`) to bold crimson (`#FF1744`).
   - The prominent banner `CRITICAL POTHOLE < 8M!` flashes at the top center.
   - Vibration motor fires (haptic alert).

---

### Test Scenario 3: Offline SQLite Queue in Dead-Zone
**Objective**: Guarantee zero data loss when mobile network connectivity drops.
1. Turn ON **Airplane Mode** on the mobile device (disabling Wi-Fi and 4G/5G).
2. Ride through simulated or physical road hazards.
3. Observe the top HUD status counter:
   `RUST FFI CORE • QUEUED SQLITE: N`
4. Confirm $N$ increments for each new hazard encountered ($N = 1, 2, 3, \dots$).
5. Attempt tapping **Sync Queue**:
   - The app safely catches network timeout without crashing.
   - The queue counter remains $N$ (records are NOT deleted upon failure).
6. **Pass Criteria**: All detections are safely held in local SQLite storage.

---

### Test Scenario 4: 3.0-Meter Spatial Deduplication Filter
**Objective**: Prevent spamming the queue when lingering near or re-detecting the same pothole across multiple video frames.
1. Stop the motorcycle within $2.0\text{m}$ of a detected pothole for 5 seconds (generating ~150 camera frames).
2. Observe the SQLite counter.
3. **Pass Criteria**:
   - Only 1 record is enqueued.
   - The subsequent 149 frames are blocked by the 3.0-meter Haversine spatial filter (`dedupRadiusMeters = 3.0`).

---

### Test Scenario 5: Reconnection & Batch Flush Ingestion
**Objective**: Confirm batch sync ingestion and backend acknowledgment.
1. Turn OFF Airplane Mode (reconnect to laptop Wi-Fi or hotspot).
2. Tap **Sync Queue** in the Flutter HUD.
3. Observe the snackbar confirmation: `Synced X hazards to backend`.
4. Check the SQLite counter: decrements by the number of acknowledged hazards (up to 50 items per batch).
5. Open the Web Console at `http://localhost:8000/console` or query `GET /api/v1/hazards/nearby`:
   - Verify that all synchronized hazards appear in the backend event log.
6. **Pass Criteria**: All $X$ hazards are synchronized, acknowledged by UUID, and cleaned from the edge phone's database.

---

## 4. Verification Matrix Summary

| Test ID | Scenario | Expected Outcome | Verification Tool |
|---|---|---|---|
| `DEV-01` | Server Ping | HTTP 200 `{"status": "healthy"}` | Mobile App Settings / `curl` |
| `DEV-02` | Horizon Alignment | Road horizon at 40% vertical height | Tactical HUD visual line |
| `DEV-03` | Critical Alert | Red banner + sound when $d < 8.0\text{m}$ | Field approach test |
| `DEV-04` | 3m Deduplication | Duplicate detections within 3m rejected | SQLite queue counter |
| `DEV-05` | Offline Durability | Detections retained across network loss | Airplane mode test |
| `DEV-06` | Batch Sync Flush | Batches of $\le 50$ committed to backend | `POST /api/v1/telemetry/sync` |
