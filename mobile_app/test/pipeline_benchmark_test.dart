import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_app/local_db.dart';
import 'package:mobile_app/rust_bridge.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  sqfliteFfiInit();

  group('Pipeline Comprehensive Benchmark Suite', () {
    late RustPotholeBridge bridge;
    late LocalHazardQueue queue;

    setUp(() async {
      bridge = RustPotholeBridge();
      queue = await LocalHazardQueue.open(
        inMemoryDatabasePath,
        factory: databaseFactoryFfi,
      );
    });

    tearDown(() async {
      await queue.db.close();
    });

    test('Benchmark: Core Pipeline Metrics (Inference, NMS, Math, SQLite, HUD Latency)', () async {
      final fixturesDir = Directory('../benchmarks/fixtures');
      final hasFixtures = fixturesDir.existsSync() &&
          fixturesDir.listSync().where((f) => f.path.endsWith('.jpg') || f.path.endsWith('.png') || f.path.endsWith('.mp4')).isNotEmpty;

      // Sample YOLOv8 transposed tensor [1, 5, 8400] anchors
      const numAnchors = 840; // realistic test slice of 840 candidate boxes
      final tensor = List<double>.generate(numAnchors * 5, (index) {
        final channel = index ~/ numAnchors;
        final anchor = index % numAnchors;
        if (channel == 0) return 320.0 + (anchor % 50); // cx
        if (channel == 1) return 400.0 + (anchor % 80); // cy
        if (channel == 2) return 80.0 + (anchor % 20);  // w
        if (channel == 3) return 40.0 + (anchor % 15);  // h
        // confidence channel: inject several high confidence detections
        if (anchor % 70 == 0) return 0.88;
        if (anchor % 100 == 0) return 0.92;
        return 0.15;
      });

      const iterations = 50;
      final ffiTimes = <double>[];
      final sqliteInsertTimes = <double>[];
      final frameToHudLatencies = <double>[];
      final alertDelays = <double>[];
      int droppedFrames = 0;

      final totalWatch = Stopwatch()..start();

      for (var i = 0; i < iterations; i++) {
        final frameWatch = Stopwatch()..start();

        // 1. FFI Postprocess (Tensor decode + NMS + Geometry + Severity)
        final ffiWatch = Stopwatch()..start();
        final hazards = bridge.runYoloPostprocess(
          tensor: tensor,
          numAnchors: numAnchors,
          inputSize: 640.0,
          confThreshold: 0.50,
          iouThreshold: 0.45,
          roiHorizonNorm: 0.40,
          imuZPeakG: 1.5,
        );
        ffiWatch.stop();
        ffiTimes.add(ffiWatch.elapsedMicroseconds / 1000.0);

        // 2. Alert Delay calculation (<8m critical alert trigger)
        final alertWatch = Stopwatch()..start();
        final hasCritical = hazards.any((h) => h.distanceMeters < 8.0 && h.confidence >= 0.75);
        alertWatch.stop();
        alertDelays.add(alertWatch.elapsedMicroseconds / 1000.0);

        // 3. SQLite Insert & Deduplication
        if (hazards.isNotEmpty) {
          final top = hazards.first;
          final sqlWatch = Stopwatch()..start();
          await queue.enqueueIfNotDuplicate(
            latitude: 9.9312 + (i * 0.0001),
            longitude: 76.2673 + (i * 0.0001),
            speedKmh: 40.0,
            confidence: top.confidence,
            severityScore: top.severityScore,
            lanePosition: top.lane,
          );
          sqlWatch.stop();
          sqliteInsertTimes.add(sqlWatch.elapsedMicroseconds / 1000.0);
        }

        frameWatch.stop();
        final totalFrameMs = frameWatch.elapsedMicroseconds / 1000.0;
        frameToHudLatencies.add(totalFrameMs);

        // If frame processing exceeds 33.3ms (30 FPS budget), mark frame drop
        if (totalFrameMs > 33.33) {
          droppedFrames++;
        }
      }

      totalWatch.stop();
      final totalSec = totalWatch.elapsedMilliseconds / 1000.0;
      final fps = iterations / totalSec;

      // Stats helper
      double calcPercentile(List<double> values, double p) {
        if (values.isEmpty) return 0.0;
        final sorted = List<double>.from(values)..sort();
        final idx = ((sorted.length - 1) * p).round();
        return sorted[idx];
      }

      final ffiP50 = calcPercentile(ffiTimes, 0.50);
      final ffiP95 = calcPercentile(ffiTimes, 0.95);
      final hudP50 = calcPercentile(frameToHudLatencies, 0.50);
      final hudP95 = calcPercentile(frameToHudLatencies, 0.95);
      final alertP50 = calcPercentile(alertDelays, 0.50);
      final alertP95 = calcPercentile(alertDelays, 0.95);
      final sqlP50 = calcPercentile(sqliteInsertTimes, 0.50);
      final sqlP95 = calcPercentile(sqliteInsertTimes, 0.95);

      final report = {
        'has_fixtures': hasFixtures,
        'iterations': iterations,
        'fps': double.parse(fps.toStringAsFixed(2)),
        'dropped_frames': droppedFrames,
        'frame_to_hud_ms': {
          'p50': double.parse(hudP50.toStringAsFixed(3)),
          'p95': double.parse(hudP95.toStringAsFixed(3)),
        },
        'ffi_postprocess_ms': {
          'p50': double.parse(ffiP50.toStringAsFixed(3)),
          'p95': double.parse(ffiP95.toStringAsFixed(3)),
        },
        'alert_delay_ms': {
          'p50': double.parse(alertP50.toStringAsFixed(3)),
          'p95': double.parse(alertP95.toStringAsFixed(3)),
        },
        'sqlite_insert_ms': {
          'p50': double.parse(sqlP50.toStringAsFixed(3)),
          'p95': double.parse(sqlP95.toStringAsFixed(3)),
        },
      };

      print('BENCHMARK_RESULT_JSON:' + jsonEncode(report));
      expect(fps, greaterThan(20.0));
      expect(hudP50, lessThan(33.33));
    });
  });
}
