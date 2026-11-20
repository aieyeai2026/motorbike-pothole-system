import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:sensors_plus/sensors_plus.dart';
import 'package:sqflite/sqflite.dart';
import 'local_db.dart';
import 'rust_bridge.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MotorbikePotholeHudApp());
}

class MotorbikePotholeHudApp extends StatelessWidget {
  final List<PotholeHazard>? initialHazards;
  final int initialQueuedCount;
  final String? initialBackendUrl;

  const MotorbikePotholeHudApp({
    super.key,
    this.initialHazards,
    this.initialQueuedCount = 0,
    this.initialBackendUrl,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Motorbike Pothole HUD',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF070B12),
      ),
      home: PotholeHudScreen(
        initialHazards: initialHazards,
        initialQueuedCount: initialQueuedCount,
        initialBackendUrl: initialBackendUrl,
      ),
    );
  }
}

class PotholeHudScreen extends StatefulWidget {
  final List<PotholeHazard>? initialHazards;
  final int initialQueuedCount;
  final String? initialBackendUrl;

  const PotholeHudScreen({
    super.key,
    this.initialHazards,
    this.initialQueuedCount = 0,
    this.initialBackendUrl,
  });

  @override
  State<PotholeHudScreen> createState() => _PotholeHudScreenState();
}

class _PotholeHudScreenState extends State<PotholeHudScreen> {
  RustPotholeBridge? _rustBridge;
  LocalHazardQueue? _localQueue;
  CameraController? _cameraController;
  Timer? _simTimer;

  // Live Sensor Streams & Telemetry State (iPhone 12)
  StreamSubscription<Position>? _positionSub;
  StreamSubscription<UserAccelerometerEvent>? _accelSub;
  StreamSubscription<GyroscopeEvent>? _gyroSub;

  double _currentLat = 9.9312;
  double _currentLon = 76.2673;
  double _currentSpeedKmh = 0.0;
  double _currentHeadingDeg = 0.0;
  double _currentZAccelG = 1.0;
  double _imuZPeakG = 1.0;
  double _rollRateDegS = 0.0;
  bool _gpsLocked = false;
  bool _imuActive = false;

  List<PotholeHazard> _activeHazards = [];
  int _queuedCount = 0;
  final bool _simMode = true;
  double _simStep = 0.0;
  bool _cameraReady = false;
  late String _backendUrl;
  String _syncStatusMessage = '';

  @override
  void initState() {
    super.initState();
    _backendUrl = widget.initialBackendUrl ?? 'http://192.168.1.9:8000';
    _activeHazards = widget.initialHazards ?? [];
    _queuedCount = widget.initialQueuedCount;
    if (widget.initialHazards == null) {
      _initNativePipeline();
    }
  }

  Future<void> _initNativePipeline() async {
    // 1. Initialize Rust C-FFI Bridge
    try {
      _rustBridge = RustPotholeBridge();
    } catch (_) {}

    // 2. Initialize Camera Feed for Live View (iPhone 12 Wide Lens)
    try {
      final cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        final backCam = cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cameras.first,
        );
        _cameraController = CameraController(
          backCam,
          ResolutionPreset.medium,
          enableAudio: false,
        );
        await _cameraController!.initialize();
        if (mounted) {
          setState(() => _cameraReady = true);
        }
      }
    } catch (_) {}

    // 3. Initialize iPhone 12 GPS & CoreLocation Telemetry
    try {
      LocationPermission perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.whileInUse || perm == LocationPermission.always) {
        try {
          final pos = await Geolocator.getCurrentPosition(
            desiredAccuracy: LocationAccuracy.bestForNavigation,
          ).timeout(const Duration(seconds: 4));
          _currentLat = pos.latitude;
          _currentLon = pos.longitude;
          _currentSpeedKmh = pos.speed > 0 ? pos.speed * 3.6 : 0.0;
          _currentHeadingDeg = pos.heading;
          _gpsLocked = true;
          if (mounted) setState(() {});
        } catch (_) {}

        _positionSub = Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.bestForNavigation,
            distanceFilter: 1, // update every 1 meter
          ),
        ).listen((Position pos) {
          if (mounted) {
            setState(() {
              _currentLat = pos.latitude;
              _currentLon = pos.longitude;
              _currentSpeedKmh = pos.speed > 0 ? pos.speed * 3.6 : 0.0;
              _currentHeadingDeg = pos.heading;
              _gpsLocked = true;
            });
          }
        });
      }
    } catch (_) {}

    // 4. Initialize iPhone 12 CoreMotion (IMU Accelerometer & Gyroscope)
    try {
      _accelSub = userAccelerometerEventStream().listen((UserAccelerometerEvent event) {
        // Vertical shock: event.z in m/s^2 converted to G's
        final shockG = 1.0 + (event.z.abs() / 9.80665);
        if (shockG > _imuZPeakG) {
          _imuZPeakG = shockG;
        } else {
          _imuZPeakG = (_imuZPeakG * 0.94) + (1.0 * 0.06);
        }
        _currentZAccelG = shockG;
        _imuActive = true;
        if (mounted) setState(() {});
      });

      _gyroSub = gyroscopeEventStream().listen((GyroscopeEvent event) {
        _rollRateDegS = event.x * 180.0 / 3.14159265;
      });
    } catch (_) {}

    // 5. Initialize SQLite Offline Queue
    try {
      final dbFolder = await getDatabasesPath();
      _localQueue = await LocalHazardQueue.open(p.join(dbFolder, 'pothole_hud.db'));
      final pending = await _localQueue!.getAllPending();
      if (mounted) {
        setState(() => _queuedCount = pending.length);
      }
    } catch (_) {}

    // 6. Start HUD Animation & Detection Loop
    _startSimulationLoop();
  }

  void _startSimulationLoop() {
    _simTimer?.cancel();
    _simTimer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
      if (!_simMode) return;
      _simStep = (_simStep + 10.0) % 240.0;
      final cy = 300.0 + _simStep;

      List<PotholeHazard> detections;
      if (_rustBridge != null) {
        final simTensor = <double>[
          320.0, 490.0,
          cy,    390.0,
          130.0, 95.0,
          68.0,  52.0,
          0.91,  0.82,
        ];
        detections = _rustBridge!.runYoloPostprocess(
          tensor: simTensor,
          numAnchors: 2,
          imuZPeakG: _imuZPeakG,
        );
      } else {
        // High-fidelity simulated detection stream fused with live iPhone 12 IMU shock
        final normY = (_simStep / 240.0);
        final dist1 = 28.0 * (1.0 - normY) + 3.8;
        final isCrit1 = dist1 < 8.0;

        // Severity dynamically increases when phone detects real physical shock/bump
        final dynamicSeverity = (isCrit1 ? 7.5 : 5.0) + ((_imuZPeakG - 1.0) * 3.0).clamp(0.0, 2.5);

        detections = [
          PotholeHazard(
            xMin: 0.42,
            yMin: (0.40 + normY * 0.44).clamp(0.0, 1.0),
            xMax: 0.58,
            yMax: (0.46 + normY * 0.48).clamp(0.0, 1.0),
            confidence: 0.93,
            distanceMeters: dist1,
            severityScore: double.parse(dynamicSeverity.toStringAsFixed(1)),
            lane: 'CENTER',
          ),
          if (dist1 > 12.0)
            const PotholeHazard(
              xMin: 0.22,
              yMin: 0.54,
              xMax: 0.34,
              yMax: 0.60,
              confidence: 0.82,
              distanceMeters: 16.2,
              severityScore: 5.6,
              lane: 'LEFT',
            ),
        ];
      }

      if (_localQueue != null && detections.isNotEmpty) {
        final top = detections.first;
        if (top.confidence >= 0.75) {
          final inserted = await _localQueue!.enqueueIfNotDuplicate(
            latitude: _currentLat,
            longitude: _currentLon,
            speedKmh: _currentSpeedKmh > 0 ? _currentSpeedKmh : 38.0,
            confidence: top.confidence,
            severityScore: top.severityScore,
            lanePosition: top.lane,
          );
          if (inserted) {
            final all = await _localQueue!.getAllPending();
            _queuedCount = all.length;
          }
        }
      }

      if (mounted) {
        setState(() => _activeHazards = detections);
      }
    });
  }

  @override
  void dispose() {
    _simTimer?.cancel();
    _cameraController?.dispose();
    _positionSub?.cancel();
    _accelSub?.cancel();
    _gyroSub?.cancel();
    super.dispose();
  }

  bool get _hasCriticalAlert =>
      _activeHazards.any((h) => h.distanceMeters < 8.0 && h.confidence >= 0.75);

  void _showBackendSettingsDialog() {
    final controller = TextEditingController(text: _backendUrl);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF161F30),
        title: const Text('Backend Server Connection', style: TextStyle(color: Color(0xFF00E5FF))),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter server URL (e.g. Wi-Fi IP or adb reverse):',
              style: TextStyle(fontSize: 13, color: Colors.white70),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: controller,
              style: const TextStyle(color: Colors.white, fontFamily: 'monospace'),
              decoration: const InputDecoration(
                hintText: 'http://192.168.1.9:8000',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Tips:\n• Same Wi-Fi: http://<HOST_IP>:8000\n• USB Cable: adb reverse tcp:8000 tcp:8000 -> http://localhost:8000\n• Android Emulator: http://10.0.2.2:8000',
              style: TextStyle(fontSize: 11, color: Colors.white54),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () async {
              try {
                final res = await http.get(Uri.parse('${controller.text}/health')).timeout(const Duration(seconds: 3));
                if (ctx.mounted) {
                  ScaffoldMessenger.of(ctx).showSnackBar(
                    SnackBar(
                      content: Text('Health check response: ${res.statusCode} ${res.body}'),
                      backgroundColor: res.statusCode == 200 ? Colors.green : Colors.red,
                    ),
                  );
                }
              } catch (e) {
                if (ctx.mounted) {
                  ScaffoldMessenger.of(ctx).showSnackBar(
                    SnackBar(content: Text('Failed to connect: $e'), backgroundColor: Colors.red),
                  );
                }
              }
            },
            child: const Text('Test Ping', style: TextStyle(color: Color(0xFF00E5FF))),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() => _backendUrl = controller.text.trim());
              Navigator.pop(ctx);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isPortrait = MediaQuery.of(context).orientation == Orientation.portrait;

    return Scaffold(
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            // 1. Live Camera Feed (if available)
            if (_cameraReady && _cameraController != null && _cameraController!.value.isInitialized)
              SizedBox.expand(
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _cameraController!.value.previewSize?.height ?? 1280,
                    height: _cameraController!.value.previewSize?.width ?? 720,
                    child: CameraPreview(_cameraController!),
                  ),
                ),
              ),

            // 2. Tactical Perspective HUD Canvas Overlay
            CustomPaint(
              key: const Key('hud_canvas'),
              painter: TacticalHudPainter(
                hazards: _activeHazards,
                animStep: _simStep,
                hasCamera: _cameraReady,
              ),
            ),

            // 3. Top Status Header & Telemetry Controls
            Positioned(
              top: 12,
              left: 16,
              right: 16,
              child: Wrap(
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: 8,
                runSpacing: 8,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.black87,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFF00E5FF)),
                    ),
                    child: Text(
                      'RUST FFI CORE • QUEUED SQLITE: $_queuedCount',
                      key: const Key('sqlite_queue_counter'),
                      style: const TextStyle(
                        color: Color(0xFF00E5FF),
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
                  if (_hasCriticalAlert)
                    Container(
                      key: const Key('critical_hazard_banner'),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFF1744),
                        borderRadius: BorderRadius.circular(10),
                        boxShadow: const [
                          BoxShadow(color: Color(0xAAFF1744), blurRadius: 12),
                        ],
                      ),
                      child: const Text(
                        '⚠️ CRITICAL POTHOLE < 8M!',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w900,
                          fontSize: 14,
                          letterSpacing: 1,
                        ),
                      ),
                    ),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.settings, color: Color(0xFF00E5FF)),
                        tooltip: 'Configure Backend Connection',
                        onPressed: _showBackendSettingsDialog,
                      ),
                      ElevatedButton(
                        onPressed: () async {
                          if (_localQueue == null) return;
                          final synced = await _localQueue!.flushBatchToBackend(
                            apiBaseUrl: _backendUrl,
                            deviceId: 'iphone-bike-01',
                          );
                          final remaining = await _localQueue!.getAllPending();
                          if (mounted) {
                            setState(() {
                              _queuedCount = remaining.length;
                              _syncStatusMessage = 'Synced $synced items to $_backendUrl';
                            });
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('Synced $synced hazards to backend'),
                                duration: const Duration(seconds: 2),
                              ),
                            );
                          }
                        },
                        child: const Text('Sync Queue'),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // 4. Portrait Mode Guidance Banner
            if (isPortrait)
              Positioned(
                top: 70,
                left: 20,
                right: 20,
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 10),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: Colors.white24),
                  ),
                  child: const Text(
                    '🏍️ Rotate to Landscape for Handlebar Cockpit View',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                ),
              ),

            // 5. iPhone 12 Live Sensory Cockpit Readout Bar (GPS, Speed, IMU Shock, Compass)
            Positioned(
              top: isPortrait ? 98 : 58,
              left: 16,
              right: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: const Color(0xDD070B12),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0x5500E5FF)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    // GPS Coordinates
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.location_on, size: 13, color: _gpsLocked ? const Color(0xFF00E676) : Colors.amber),
                        const SizedBox(width: 4),
                        Text(
                          '${_currentLat.toStringAsFixed(4)}°, ${_currentLon.toStringAsFixed(4)}°',
                          style: TextStyle(
                            color: _gpsLocked ? const Color(0xFF00E676) : Colors.amber,
                            fontSize: 11,
                            fontFamily: 'monospace',
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    // Live Speedometer (from GPS ground speed)
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.speed, size: 13, color: Color(0xFF00E5FF)),
                        const SizedBox(width: 4),
                        Text(
                          '${_currentSpeedKmh.toStringAsFixed(0)} km/h',
                          style: const TextStyle(
                            color: Color(0xFF00E5FF),
                            fontSize: 11,
                            fontFamily: 'monospace',
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    // IMU Vertical G-Force Shock (CoreMotion Z-axis)
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.bolt,
                          size: 13,
                          color: _currentZAccelG > 1.25 ? const Color(0xFFFF1744) : const Color(0xFF00E5FF),
                        ),
                        const SizedBox(width: 3),
                        Text(
                          '${_currentZAccelG.toStringAsFixed(2)}G',
                          style: TextStyle(
                            color: _currentZAccelG > 1.25 ? const Color(0xFFFF1744) : const Color(0xFF00E5FF),
                            fontSize: 11,
                            fontFamily: 'monospace',
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    // Compass Bearing
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.explore, size: 13, color: Colors.white70),
                        const SizedBox(width: 3),
                        Text(
                          '${_currentHeadingDeg.toStringAsFixed(0)}°',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 11,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            // 6. Road Centering & Ego Reticle Badge at Bottom Center
            Positioned(
              bottom: 16,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xDD070B12),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFF00E5FF)),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text('L: 1.8m  ', style: TextStyle(color: Color(0xFF00E5FF), fontSize: 12, fontFamily: 'monospace')),
                      Text(
                        'CENTERED (0.0m)',
                        style: TextStyle(
                          color: Colors.black,
                          backgroundColor: Color(0xFF00E676),
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                      Text('  R: 1.8m', style: TextStyle(color: Color(0xFF00E5FF), fontSize: 12, fontFamily: 'monospace')),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class TacticalHudPainter extends CustomPainter {
  final List<PotholeHazard> hazards;
  final double animStep;
  final bool hasCamera;

  TacticalHudPainter({
    required this.hazards,
    this.animStep = 0.0,
    this.hasCamera = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final width = size.width;
    final height = size.height;
    final horizonY = height * 0.40;

    // 1. Synthetic Perspective Road Surface (if camera feed is not running)
    if (!hasCamera) {
      // Sky
      final skyPaint = Paint()..color = const Color(0xFF090F1C);
      canvas.drawRect(Rect.fromLTWH(0, 0, width, horizonY), skyPaint);

      // Road Surface
      final roadTopLeft = width * 0.42;
      final roadTopRight = width * 0.58;
      final roadPath = Path()
        ..moveTo(roadTopLeft, horizonY)
        ..lineTo(roadTopRight, horizonY)
        ..lineTo(width, height)
        ..lineTo(0, height)
        ..close();

      final asphaltPaint = Paint()..color = const Color(0xFF060B14);
      canvas.drawPath(roadPath, asphaltPaint);

      // Glowing Cyan Curbs / Edges
      final curbPaint = Paint()
        ..color = const Color(0xFF00E5FF)
        ..strokeWidth = 2.0;
      canvas.drawLine(Offset(roadTopLeft, horizonY), Offset(0, height), curbPaint);
      canvas.drawLine(Offset(roadTopRight, horizonY), Offset(width, height), curbPaint);
    }

    // 2. Horizon Line (40% Calibration Mark)
    final horizonPaint = Paint()
      ..color = const Color(0x6600E5FF)
      ..strokeWidth = 1.5;
    canvas.drawLine(Offset(0, horizonY), Offset(width, horizonY), horizonPaint);

    final roadTopLeft = width * 0.42;
    final roadTopRight = width * 0.58;

    // 3. Perspective Lane Dividers
    final lanePaint = Paint()
      ..color = const Color(0x4400E5FF)
      ..strokeWidth = 1.5;

    // Left Lane line
    final leftTop = roadTopLeft + (roadTopRight - roadTopLeft) * 0.33;
    canvas.drawLine(Offset(leftTop, horizonY), Offset(width * 0.33, height), lanePaint);

    // Right Lane line
    final rightTop = roadTopLeft + (roadTopRight - roadTopLeft) * 0.66;
    canvas.drawLine(Offset(rightTop, horizonY), Offset(width * 0.66, height), lanePaint);

    // 4. Motorcycle Steering Trajectory Guide
    final bikeX = width * 0.50;
    final bikeY = height - 24.0;
    final trajectoryPaint = Paint()
      ..color = const Color(0x6600E676)
      ..strokeWidth = 2.0;
    canvas.drawLine(Offset(bikeX, bikeY), Offset(width * 0.50, horizonY), trajectoryPaint);

    // 5. Motorcycle Center Reticle
    final reticlePaint = Paint()..color = const Color(0xFF00E676);
    canvas.drawCircle(Offset(bikeX, bikeY), 8.0, reticlePaint);
    final reticleRingPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawCircle(Offset(bikeX, bikeY), 8.0, reticleRingPaint);

    // 6. Hazard Bounding Boxes & Distance Overlays
    for (final h in hazards) {
      final isCritical = h.distanceMeters < 8.0;
      final color = isCritical ? const Color(0xFFFF1744) : const Color(0xFF00E5FF);

      final rect = Rect.fromLTRB(
        h.xMin * width,
        h.yMin * height,
        h.xMax * width,
        h.yMax * height,
      );

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = isCritical ? 3.5 : 2.0;
      canvas.drawRect(rect, boxPaint);

      final label = '${h.distanceMeters.toStringAsFixed(1)}m [${h.lane}]';
      final textSpan = TextSpan(
        text: label,
        style: TextStyle(
          color: color,
          fontSize: isCritical ? 13 : 11,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black87,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(canvas, Offset(rect.left, (rect.top - 18).clamp(0.0, height - 20)));
    }
  }

  @override
  bool shouldRepaint(covariant TacticalHudPainter oldDelegate) => true;
}
