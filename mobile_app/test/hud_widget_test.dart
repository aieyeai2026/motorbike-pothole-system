import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_app/main.dart';
import 'package:mobile_app/rust_bridge.dart';

void main() {
  testWidgets('T-2.3.1: HUD renders critical alert banner when Rust pothole is < 8m', (tester) async {
    const closeHazard = PotholeHazard(
      xMin: 0.40,
      yMin: 0.75,
      xMax: 0.60,
      yMax: 0.92,
      confidence: 0.91,
      distanceMeters: 4.5,
      severityScore: 8.2,
      lane: 'CENTER',
    );

    await tester.pumpWidget(
      const MotorbikePotholeHudApp(
        initialHazards: [closeHazard],
        initialQueuedCount: 3,
      ),
    );

    expect(find.byKey(const Key('hud_canvas')), findsOneWidget);
    expect(find.byKey(const Key('critical_hazard_banner')), findsOneWidget);
    expect(find.textContaining('QUEUED SQLITE: 3'), findsOneWidget);
  });

  testWidgets('T-2.3.2: HUD renders live iPhone 12 telemetry sensory gauges', (tester) async {
    await tester.pumpWidget(
      const MotorbikePotholeHudApp(
        initialQueuedCount: 0,
      ),
    );

    expect(find.byIcon(Icons.location_on), findsOneWidget);
    expect(find.byIcon(Icons.speed), findsOneWidget);
    expect(find.byIcon(Icons.bolt), findsOneWidget);
    expect(find.byIcon(Icons.explore), findsOneWidget);
  });

  testWidgets('T-4.4.1: HUD alerts only for pothole < 8m, not when only car is < 8m', (tester) async {
    const closeCarOnly = PotholeHazard(
      xMin: 0.40,
      yMin: 0.70,
      xMax: 0.60,
      yMax: 0.90,
      confidence: 0.93,
      distanceMeters: 5.0,
      severityScore: 6.5,
      lane: 'CENTER',
      classId: 1, // 1 = car
    );

    await tester.pumpWidget(
      const MotorbikePotholeHudApp(
        initialHazards: [closeCarOnly],
        initialQueuedCount: 0,
      ),
    );

    expect(find.byKey(const Key('critical_hazard_banner')), findsNothing);
  });

  test('T-5.2.1: computeTargetFps enforces speed-adaptive and thermal throttling', () {
    expect(computeTargetFps(speedKmh: 0.0), 0);
    expect(computeTargetFps(speedKmh: 22.0), 10);
    expect(computeTargetFps(speedKmh: 55.0), 30);
    expect(computeTargetFps(speedKmh: 55.0, thermalModerate: true), 10);
  });

  testWidgets('T-6.2.1: Phase 6 HUD alerts only when pothole < 10m is inside trajectory path', (tester) async {
    const outOfPathPothole = PotholeHazard(
      xMin: 0.15,
      yMin: 0.72,
      xMax: 0.25,
      yMax: 0.80,
      confidence: 0.91,
      distanceMeters: 6.2,
      severityScore: 7.4,
      lane: 'LEFT',
      classId: 0,
      inTrajectoryPath: false,
    );

    await tester.pumpWidget(
      const MotorbikePotholeHudApp(
        initialHazards: [outOfPathPothole],
        initialQueuedCount: 1,
      ),
    );
    expect(find.byKey(const Key('critical_hazard_banner')), findsNothing);

    const inPathPothole = PotholeHazard(
      xMin: 0.45,
      yMin: 0.74,
      xMax: 0.55,
      yMax: 0.82,
      confidence: 0.92,
      distanceMeters: 6.4,
      severityScore: 8.1,
      lane: 'CENTER',
      classId: 0,
      inTrajectoryPath: true,
    );

    await tester.pumpWidget(
      const MotorbikePotholeHudApp(
        initialHazards: [inPathPothole],
        initialQueuedCount: 1,
      ),
    );
    expect(find.byKey(const Key('critical_hazard_banner')), findsOneWidget);
    expect(find.textContaining('POTHOLE IN PATH: 6.4m'), findsOneWidget);
  });
}
