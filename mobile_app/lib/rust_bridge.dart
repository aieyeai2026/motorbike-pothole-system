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

  const PotholeHazard({
    required this.xMin,
    required this.yMin,
    required this.xMax,
    required this.yMax,
    required this.confidence,
    required this.distanceMeters,
    required this.severityScore,
    required this.lane,
  });

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

  RustPotholeBridge({String? overrideLibraryPath}) {
    _lib = _openLibrary(overrideLibraryPath);
    _runPostprocess = _lib
        .lookup<ffi.NativeFunction<_RunPostprocessNative>>(
          'pothole_core_run_postprocess',
        )
        .asFunction<_RunPostprocessDart>();
  }

  static ffi.DynamicLibrary _openLibrary(String? overridePath) {
    if (overridePath != null) {
      return ffi.DynamicLibrary.open(overridePath);
    }
    if (Platform.isAndroid) {
      return ffi.DynamicLibrary.open('libpothole_core.so');
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
