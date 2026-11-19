import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_app/rust_bridge.dart';

void main() {
  test('T-2.1.2: RustPotholeBridge FFI decodes YOLOv8 tensor and maps CENTER lane', () {
    final bridge = RustPotholeBridge();
    final tensor = <double>[
      320.0, 324.0, 100.0,
      480.0, 482.0, 200.0,
      128.0, 130.0, 40.0,
      64.0,  66.0,  40.0,
      0.89,  0.74,  0.10,
    ];

    final hazards = bridge.runYoloPostprocess(
      tensor: tensor,
      numAnchors: 3,
      inputSize: 640.0,
      confThreshold: 0.50,
      iouThreshold: 0.45,
      roiHorizonNorm: 0.40,
      imuZPeakG: 1.6,
    );

    expect(hazards.length, 1);
    expect(hazards.first.lane, 'CENTER');
    expect(hazards.first.confidence, closeTo(0.89, 1e-4));
    expect(hazards.first.distanceMeters, greaterThanOrEqualTo(2.0));
  });
}
