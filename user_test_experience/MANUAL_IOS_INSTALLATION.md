# Motorbike Pothole System — Apple Device (iOS) Installation & Testing Manual

This manual provides a complete, step-by-step guide to installing, configuring, and running the Motorbike Pothole Detection & Tactical HUD system on Apple devices (**iPhone** and **iPad**).

---

## 2 Ways to Run on Apple Devices

Depending on your setup, choose the method that fits best:

| Option | Requirements | Setup Time | Features Supported |
|---|---|---|---|
| **Option 1: Safari Web HUD (Fastest)** | Any iPhone with Safari (Zero Install) | **30 seconds** | Tactical HUD, Live Camera Feed, Audio Alarms, Telemetry Sync |
| **Option 2: Native Flutter App** | Mac with Xcode & USB Cable | **10 minutes** | Native Camera Feed, Rust C-ABI FFI, Offline SQLite Queue, Haptic Feedback |

---

## Option 1: Instant Safari Web App (Zero Install — Recommended for Quick Field Testing)

You do **not** need Xcode, a paid Apple Developer account, or any software installation on your iPhone for this option.

### Step 1: Start the Backend on Your Mac
In your terminal, run:
```bash
python user_test_experience/start_test_server.py
```
Note your Mac's local Wi-Fi IP printed in the terminal (e.g. `http://192.168.1.9:8000`).

### Step 2: Open Safari on iPhone
1. Connect your iPhone to the **same Wi-Fi network** as your Mac.
2. Open **Safari** (not Chrome or third-party browsers, as Safari has native iOS camera permissions).
3. Navigate to:
   ```text
   http://192.168.1.9:8000/console/
   ```
4. When prompted by Safari, tap **Allow** for Camera and Motion sensors.

### Step 3: Add to Home Screen (Full Screen App Mode)
1. In Safari, tap the **Share** button (box with an upward arrow at the bottom).
2. Scroll down and tap **Add to Home Screen**.
3. Name it **"Pothole HUD"** and tap **Add**.
4. Open the new icon from your home screen: it runs in immersive full-screen landscape mode without browser navigation bars!

---

## Option 2: Native Flutter iOS App Deployment

Follow these steps to compile and install the native Flutter app with the Rust edge vision core onto a physical iPhone.

### Prerequisites on Your Mac
- macOS machine with [Flutter SDK](https://flutter.dev/docs/get-started/install/macos) installed.
- [Xcode](https://apps.apple.com/us/app/xcode/id497799835) installed from the Mac App Store.
- CocoaPods (`brew install cocoapods` or `sudo gem install cocoapods`).
- Free Apple ID account.

---

### Step 1: Enable Developer Mode on iPhone (iOS 16, 17, 18)
1. On your iPhone, open **Settings**.
2. Go to **Privacy & Security** -> scroll down to **Developer Mode**.
3. Toggle **Developer Mode ON**.
4. Tap **Restart**.
5. After reboot, unlock your phone and tap **Turn On** when prompted.

---

### Step 2: Build the Rust Edge Vision Core for iOS
In your Mac terminal:
```bash
# Add the 64-bit ARM iOS target
rustup target add aarch64-apple-ios

# Build optimized release binary
cd /Users/gautam/motorbike-pothole-system/pothole_core
cargo build --target aarch64-apple-ios --release
```
The compiled static library is output at:
`pothole_core/target/aarch64-apple-ios/release/libpothole_core.a`

---

### Step 3: Configure Xcode Signing & Permissions
1. Connect your iPhone to your Mac via USB cable.
2. Tap **Trust This Computer** on your iPhone and enter your passcode.
3. Open the iOS project in Xcode:
   ```bash
   cd /Users/gautam/motorbike-pothole-system/mobile_app
   open ios/Runner.xcworkspace
   ```
4. In Xcode's left sidebar, select the top-level **Runner** project.
5. In the center pane, click on the **Signing & Capabilities** tab:
   - Check **Automatically manage signing**.
   - Under **Team**, sign in with your Apple ID and select your personal team (`<Your Name> (Personal Team)`).
   - In **Bundle Identifier**, append your name to make it unique (e.g. `com.aieyeai.motorbikePotholeHud`).
6. Confirm the permissions configured in `Runner/Info.plist`:
   - `NSCameraUsageDescription` (Camera access for road detection)
   - `NSLocationWhenInUseUsageDescription` (GPS coordinates for pothole tagging)
   - `NSLocalNetworkUsageDescription` (Connecting to your Mac over Wi-Fi)
   - `NSAppTransportSecurity` (Allows connecting to local HTTP `http://192.168.1.9:8000`)

---

### Step 4: Run the App on Your iPhone
In your terminal:
```bash
cd /Users/gautam/motorbike-pothole-system/mobile_app

# List connected devices and get your iPhone's identifier
flutter devices

# Deploy and launch the app
flutter run -d <your-iphone-device-id>
```

#### First-time App Verification on iPhone:
If iOS shows an "Untrusted Developer" dialog when the app first opens:
1. Open **Settings** on your iPhone.
2. Go to **General** -> **VPN & Device Management**.
3. Tap your Apple ID under **Developer App**.
4. Tap **Trust "<Your Apple ID>"** -> **Trust**.
5. Reopen the app from your iPhone's home screen.

---

### Step 5: Connect iPhone to Telemetry Backend & Field Test

1. Ensure your Mac server is running:
   ```bash
   python user_test_experience/start_test_server.py
   ```
2. Open the **Motorbike Pothole HUD** app on your iPhone.
3. Tap the **Settings (gear)** icon in the top header.
4. Enter your Mac's Wi-Fi IP address:
   `http://192.168.1.9:8000`
5. Tap **Test Ping**:
   - A green notification confirms: `Health check response: 200 {"status": "healthy"}`.
6. Tap **Save**.

---

## 3. Physical Motorcycle Mount for iPhone

> [!WARNING]
> **VIBRATION DAMPENER REQUIRED FOR iPHONES**:
> Modern iPhones (iPhone 11 through iPhone 16) feature delicate **Sensor-Shift Optical Image Stabilization (OIS)** magnets. Direct exposure to single-cylinder motorcycle engine harmonic vibrations can damage the camera autofocus.
> **Always use an active vibration dampening adapter** (e.g. *QuadLock Motorcycle Vibration Dampener* or *SP Connect Anti-Vibration Module*).

1. **Mount Position**: Center of handlebars or fork stem mount, orienting the iPhone in **Landscape Mode**.
2. **Camera Alignment**: Ensure the camera lens has an unobstructed view over the windshield/fairing. Align the road horizon with the 44% horizontal cyan guideline on the HUD.
3. **Power**: Keep connected to a 5V/2.4A USB charger on the bike to maintain continuous 30 FPS camera inference without draining the battery.
4. **Field Verification**:
   - When approaching potholes within $8.0\text{m}$, the crimson alert banner flashes with audio alert.
   - In low-connectivity areas, turn on **Airplane Mode** to verify that detections accumulate safely in your offline SQLite queue without data loss.
