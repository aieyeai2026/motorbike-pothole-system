import 'dart:async';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
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
  Timer? _simTimer;

  List<PotholeHazard> _activeHazards = [];
  int _queuedCount = 0;
  final bool _simMode = true;
  double _simStep = 0.0;
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
    try {
      _rustBridge = RustPotholeBridge();
    } catch (_) {}

    try {
      final dbFolder = await getDatabasesPath();
      _localQueue = await LocalHazardQueue.open(p.join(dbFolder, 'pothole_hud.db'));
      final pending = await _localQueue!.getAllPending();
      if (mounted) {
        setState(() => _queuedCount = pending.length);
      }
    } catch (_) {}

    _startSimulationLoop();
  }

  void _startSimulationLoop() {
    _simTimer?.cancel();
    _simTimer = Timer.periodic(const Duration(milliseconds: 120), (_) async {
      if (!_simMode || _rustBridge == null) return;
      _simStep = (_simStep + 12.0) % 220.0;
      final cy = 320.0 + _simStep;

      final simTensor = <double>[
        320.0, 490.0,
        cy,    390.0,
        130.0, 95.0,
        68.0,  52.0,
        0.91,  0.82,
      ];

      final detections = _rustBridge!.runYoloPostprocess(
        tensor: simTensor,
        numAnchors: 2,
        imuZPeakG: 1.4,
      );

      if (_localQueue != null && detections.isNotEmpty) {
        final top = detections.first;
        if (top.confidence >= 0.75) {
          final inserted = await _localQueue!.enqueueIfNotDuplicate(
            latitude: 9.9312 + (_simStep * 0.00005),
            longitude: 76.2673,
            speedKmh: 48.0,
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
    return Scaffold(
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            CustomPaint(
              key: const Key('hud_canvas'),
              painter: TacticalHudPainter(hazards: _activeHazards),
            ),
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
                      ),
                      child: const Text(
                        'CRITICAL POTHOLE < 8M!',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w900,
                          fontSize: 14,
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
                            deviceId: 'android-bike-01',
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
          ],
        ),
      ),
    );
  }
}

class TacticalHudPainter extends CustomPainter {
  final List<PotholeHazard> hazards;

  TacticalHudPainter({required this.hazards});

  @override
  void paint(Canvas canvas, Size size) {
    final horizonY = size.height * 0.40;
    final horizonPaint = Paint()
      ..color = const Color(0x5500E5FF)
      ..strokeWidth = 1.5;
    canvas.drawLine(Offset(0, horizonY), Offset(size.width, horizonY), horizonPaint);

    for (final h in hazards) {
      final isCritical = h.distanceMeters < 8.0;
      final color = isCritical ? const Color(0xFFFF1744) : const Color(0xFF00E5FF);

      final rect = Rect.fromLTRB(
        h.xMin * size.width,
        h.yMin * size.height,
        h.xMax * size.width,
        h.yMax * size.height,
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
          fontSize: isCritical ? 14 : 12,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black87,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top - 18));
    }
  }

  @override
  bool shouldRepaint(covariant TacticalHudPainter oldDelegate) => true;
}
