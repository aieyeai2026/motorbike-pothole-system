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

  test('T-4.2.1: RustPotholeBridge decodes multi-class tensor and computes lane centering', () {
    final bridge = RustPotholeBridge();
    final tensor = <double>[
      320.0, 322.0, 150.0, // cx
      480.0, 481.0, 500.0, // cy
      120.0, 120.0,  60.0, // w
       60.0,  60.0,  40.0, // h
       0.92,  0.10,  0.05, // class 0: pothole
       0.08,  0.86,  0.10, // class 1: car
       0.05,  0.08,  0.79, // class 2: debris
    ];

    final hazards = bridge.runYoloPostprocess(
      tensor: tensor,
      numAnchors: 3,
    );

    expect(hazards.length, 3);
    expect(hazards[0].hazardType, 'pothole');
    expect(hazards[1].hazardType, 'car');
    expect(hazards[2].hazardType, 'debris');

    final centering = bridge.computeLaneCentering(
      xEgo: 0.80,
      xLeftBorder: 0.0,
      xRightBorder: 1.0,
      roadWidthM: 3.6,
    );
    expect(centering.status, 'DRIFTING_RIGHT');
    expect(centering.offsetMeters, greaterThan(0.25));
  });
}
