import 'dart:ffi' as ffi;
import 'dart:io' show Platform, Directory;
import 'package:ffi/ffi.dart';

final class CDetectedPothole extends ffi.Struct {
  @ffi.Float()
  external double xMin;

  @ffi.Float()
  external double yMin;

  @ffi.Float()
  external double xMax;

  @ffi.Float()
  external double yMax;

  @ffi.Float()
  external double confidence;

  @ffi.Float()
  external double distanceMeters;

  @ffi.Float()
  external double severityScore;

  @ffi.Int32()
  external int laneCode;

  @ffi.Int32()
  external int classId;
}

final class CLaneCentering extends ffi.Struct {
  @ffi.Float()
  external double offsetMeters;

  @ffi.Float()
  external double distLeftM;

  @ffi.Float()
  external double distRightM;

  @ffi.Int32()
  external int statusCode;
}

typedef _ComputeLaneCenteringNative = ffi.Int32 Function(
  ffi.Float xEgo,
  ffi.Float xLeftBorder,
  ffi.Float xRightBorder,
  ffi.Float roadWidthM,
  ffi.Pointer<CLaneCentering> outPtr,
);

typedef _ComputeLaneCenteringDart = int Function(
  double xEgo,
  double xLeftBorder,
  double xRightBorder,
  double roadWidthM,
  ffi.Pointer<CLaneCentering> outPtr,
);

class LaneCenteringResult {
  final double offsetMeters;
  final double distLeftM;
  final double distRightM;
  final String status;

  const LaneCenteringResult({
    required this.offsetMeters,
    required this.distLeftM,
    required this.distRightM,
    required this.status,
  });

  static String statusFromCode(int code) {
    switch (code) {
      case 0:
        return 'CENTERED';
      case 1:
        return 'DRIFTING_LEFT';
      case 2:
        return 'DRIFTING_RIGHT';
      default:
        return 'CENTERED';
    }
  }
}

typedef _RunPostprocessNative = ffi.Size Function(
  ffi.Pointer<ffi.Float> tensorPtr,
  ffi.Size tensorLen,
  ffi.Size numAnchors,
  ffi.Float inputSize,
  ffi.Float confThreshold,
  ffi.Float iouThreshold,
  ffi.Float roiHorizonNorm,
  ffi.Float imuZPeakG,
  ffi.Pointer<CDetectedPothole> outBufferPtr,
  ffi.Size maxOut,
);

typedef _RunPostprocessDart = int Function(
  ffi.Pointer<ffi.Float> tensorPtr,
  int tensorLen,
  int numAnchors,
  double inputSize,
  double confThreshold,
  double iouThreshold,
  double roiHorizonNorm,
  double imuZPeakG,
  ffi.Pointer<CDetectedPothole> outBufferPtr,
  int maxOut,
);

class PotholeHazard {
  final double xMin;
  final double yMin;
  final double xMax;
  final double yMax;
  final double confidence;
  final double distanceMeters;
  final double severityScore;
  final String lane;
  final int classId;

  const PotholeHazard({
    required this.xMin,
    required this.yMin,
    required this.xMax,
    required this.yMax,
    required this.confidence,
    required this.distanceMeters,
    required this.severityScore,
    required this.lane,
    this.classId = 0,
  });

  String get hazardType {
    switch (classId) {
      case 0:
        return 'pothole';
      case 1:
        return 'car';
      case 2:
        return 'debris';
      default:
        return 'pothole';
    }
  }

  static String laneFromCode(int code) {
    switch (code) {
      case 0:
        return 'LEFT';
      case 1:
        return 'CENTER';
      case 2:
        return 'RIGHT';
      default:
        return 'CENTER';
    }
  }
}

class RustPotholeBridge {
  late final ffi.DynamicLibrary _lib;
  late final _RunPostprocessDart _runPostprocess;
  late final _ComputeLaneCenteringDart _computeLaneCentering;

  RustPotholeBridge({String? overrideLibraryPath}) {
    _lib = _openLibrary(overrideLibraryPath);
    _runPostprocess = _lib
        .lookup<ffi.NativeFunction<_RunPostprocessNative>>(
          'pothole_core_run_postprocess',
        )
        .asFunction<_RunPostprocessDart>();
    _computeLaneCentering = _lib
        .lookup<ffi.NativeFunction<_ComputeLaneCenteringNative>>(
          'pothole_core_compute_lane_centering',
        )
        .asFunction<_ComputeLaneCenteringDart>();
  }

  LaneCenteringResult computeLaneCentering({
    required double xEgo,
    required double xLeftBorder,
    required double xRightBorder,
    double roadWidthM = 3.6,
  }) {
    final outPtr = calloc<CLaneCentering>();
    try {
      _computeLaneCentering(xEgo, xLeftBorder, xRightBorder, roadWidthM, outPtr);
      final ref = outPtr.ref;
      return LaneCenteringResult(
        offsetMeters: ref.offsetMeters,
        distLeftM: ref.distLeftM,
        distRightM: ref.distRightM,
        status: LaneCenteringResult.statusFromCode(ref.statusCode),
      );
    } finally {
      calloc.free(outPtr);
    }
  }

  static ffi.DynamicLibrary _openLibrary(String? overridePath) {
    if (overridePath != null) {
      return ffi.DynamicLibrary.open(overridePath);
    }
    if (Platform.isAndroid) {
      return ffi.DynamicLibrary.open('libpothole_core.so');
    }
    if (Platform.isIOS) {
      // In iOS, native symbols from static libraries or embedded frameworks are resolved via process()
      return ffi.DynamicLibrary.process();
    }
    if (Platform.isMacOS) {
      final releasePath =
          '${Directory.current.parent.path}/pothole_core/target/release/libpothole_core.dylib';
      return ffi.DynamicLibrary.open(releasePath);
    }
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  List<PotholeHazard> runYoloPostprocess({
    required List<double> tensor,
    required int numAnchors,
    double inputSize = 640.0,
    double confThreshold = 0.50,
    double iouThreshold = 0.45,
    double roiHorizonNorm = 0.40,
    double imuZPeakG = 1.0,
    int maxOut = 20,
  }) {
    final tensorPtr = calloc<ffi.Float>(tensor.length);
    final outPtr = calloc<CDetectedPothole>(maxOut);

    try {
      for (var i = 0; i < tensor.length; i++) {
        tensorPtr[i] = tensor[i];
      }

      final count = _runPostprocess(
        tensorPtr,
        tensor.length,
        numAnchors,
        inputSize,
        confThreshold,
        iouThreshold,
        roiHorizonNorm,
        imuZPeakG,
        outPtr,
        maxOut,
      );

      final results = <PotholeHazard>[];
      for (var i = 0; i < count; i++) {
        final item = (outPtr + i).ref;
        results.add(
          PotholeHazard(
            xMin: item.xMin,
            yMin: item.yMin,
            xMax: item.xMax,
            yMax: item.yMax,
            confidence: item.confidence,
            distanceMeters: item.distanceMeters,
            severityScore: item.severityScore,
            lane: PotholeHazard.laneFromCode(item.laneCode),
            classId: item.classId,
          ),
        );
      }
      return results;
    } finally {
      calloc.free(tensorPtr);
      calloc.free(outPtr);
    }
  }
}
