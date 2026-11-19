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
}
