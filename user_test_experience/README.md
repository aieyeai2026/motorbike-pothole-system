# User Test Experience & Device Testing Suite

Interactive environment for connecting, testing, and verifying the Motorbike Pothole System with physical mobile devices and simulated riders.

---

## Quick Start (3 Steps)

### 1. Launch the Backend Server
```bash
python user_test_experience/start_test_server.py
```
This runs the FastAPI backend on `0.0.0.0:8000`, automatically discovers your computer's local Wi-Fi IP (e.g. `192.168.1.9`), and hosts the interactive Tactical Test Console.

### 2. Connect Your Mobile Device
- **Physical Mobile Phone (Same Wi-Fi)**:
  Open your mobile browser or the Flutter app (`mobile_app`), tap the Settings icon, and point to `http://<YOUR_LAN_IP>:8000`.
- **Physical Mobile Phone (USB Cable)**:
  Run `adb reverse tcp:8000 tcp:8000` and point the mobile app to `http://localhost:8000`.
- **Desktop / Laptop Web Console**:
  Open [http://127.0.0.1:8000/console](http://127.0.0.1:8000/console) in any browser.

### 3. Run Simulated Telemetry Test
In a separate terminal:
```bash
python user_test_experience/simulate_ride.py --server http://127.0.0.1:8000 --num-hazards 5
```

---

## Detailed Manual for Physical Device Testing
See **[MANUAL_DEVICE_TESTING.md](MANUAL_DEVICE_TESTING.md)** for:
- Motorbike handlebar & vibration dampener mounting guide.
- Camera 40% horizon calibration procedures.
- 5 step-by-step road test scenarios (<8m alerts, dead-zone offline SQLite queueing, 3m spatial deduplication, batch syncing).

---

## Component Directory
- `MANUAL_DEVICE_TESTING.md`: Complete field test procedure for physical motorbike and smartphone testing.
- `start_test_server.py`: Network launcher with LAN IP auto-detection and mobile setup guide.
- `simulate_ride.py`: CLI telemetry client simulator generating realistic coordinates, IMU data, and batch syncing.
- `web_console/`: Interactive Tactical Motorcycle HUD simulator, real-time alert trigger, and telemetry sync console.
  - `index.html`: Dashboard UI.
  - `style.css`: Cyberpunk tactical HUD styling.
  - `app.js`: Web Audio API alert sound, Canvas HUD renderer, and sync client.
- `tests/test_user_experience.py`: Automated test suite for user test experience features.
