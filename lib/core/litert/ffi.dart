import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as p;

/// Opaque `LiteRt*` handle (environment, model, compiled model, ...).
typedef Handle = Pointer<Void>;

/// Non-OK `LiteRtStatus` returned by the C API.
class LiteRtException implements Exception {
  LiteRtException(this.call, this.status, this.statusName);
  final String call;
  final int status;
  final String statusName;

  @override
  String toString() => 'LiteRT $call failed: $statusName ($status)';
}

// Element types (litert_model_types.h).
const kElementFloat32 = 1;
const kElementInt32 = 2;
const kElementUInt8 = 3;
const kElementInt64 = 4;
const kElementBool = 6;
const kElementInt16 = 7;
const kElementInt8 = 9;
const kElementFloat16 = 10;

// LiteRtHwAccelerators bit flags (litert_common.h).
const kHwCpu = 1;
const kHwGpu = 2;
const kHwNpu = 4;

// LiteRtTensorBufferType (litert_tensor_buffer_types.h).
const kBufferHostMemory = 1;

// LiteRtTensorBufferLockMode.
const kLockRead = 0;
const kLockWrite = 1;

// LiteRtEnvOptionTag (litert_environment_options.h).
const kEnvRuntimeLibraryDir = 22;

// LiteRtAnyType.
const kAnyString = 8;

/// Byte layout of `LiteRtRankedTensorType` (element type + `LiteRtLayout`).
const rankedTensorTypeBytes = 72;
const _layoutOffset = 4;
const maxTensorRank = 8;

/// Byte layout of `LiteRtEnvOption` { tag, LiteRtAny { type, union } }.
const envOptionBytes = 24;

/// Reads the element type of a `LiteRtRankedTensorType` at [t].
int rankedElementType(Pointer<Uint8> t) => t.cast<Int32>().value;

/// Reads the dimensions of a `LiteRtRankedTensorType` at [t].
List<int> rankedShape(Pointer<Uint8> t) {
  final layout = (t + _layoutOffset);
  final rank = layout.value & 0x7f; // 7-bit bitfield
  final dims = (layout + 4).cast<Int32>();
  return [for (var i = 0; i < rank; i++) dims[i]];
}

typedef _H = Pointer<Void>;
typedef _HH = Pointer<Pointer<Void>>;

/// Thin typed view of the LiteRT C API (`libLiteRt`), resolved lazily.
///
/// Only the CompiledModel path is bound; see `litert_cc_sdk/litert/c` for the
/// headers (fetch them with `py -I tools/fetch_litert_sdk.py`). The library
/// exports the C API directly, so no native glue code is compiled.
class LiteRtApi {
  LiteRtApi._(this.library, this.libraryPath);

  final DynamicLibrary library;

  /// Absolute path the library was loaded from (or its bare name).
  final String libraryPath;

  static LiteRtApi? _instance;
  static LiteRtApi get instance => _instance ??= _load();

  static String get libraryFileName => switch (Platform.operatingSystem) {
    'windows' => 'libLiteRt.dll',
    'macos' || 'ios' => 'libLiteRt.dylib',
    _ => 'libLiteRt.so',
  };

  static String? get _devPlatformDir => switch (Platform.operatingSystem) {
    'windows' => 'windows_x86_64',
    'macos' => 'macos_arm64',
    'android' => 'android_arm64',
    _ => null,
  };

  /// Directories searched for [libraryFileName], in order: next to the
  /// executable (Windows/Linux bundle), the macOS app's `Frameworks`, then a
  /// `litert_runtime/<platform>` checkout folder (dev runs and tests).
  static List<String> searchDirs() {
    final exeDir = p.dirname(Platform.resolvedExecutable);
    final dirs = <String>[
      exeDir,
      p.normalize(p.join(exeDir, '..', 'Frameworks')),
      p.join(exeDir, 'lib'),
    ];
    final platform = _devPlatformDir;
    if (platform != null) {
      for (final start in [Directory.current.path, exeDir]) {
        var dir = Directory(start);
        for (var i = 0; i < 8; i++) {
          dirs.add(p.join(dir.path, 'litert_runtime', platform));
          final parent = dir.parent;
          if (parent.path == dir.path) break;
          dir = parent;
        }
      }
    }
    return dirs;
  }

  static LiteRtApi _load() {
    final name = libraryFileName;
    for (final dir in searchDirs()) {
      final path = p.join(dir, name);
      if (File(path).existsSync()) {
        return LiteRtApi._(DynamicLibrary.open(path), path);
      }
    }
    // Android resolves bare names from the APK's jniLibs; elsewhere this
    // falls back to the OS search path and throws a clear error if absent.
    try {
      return LiteRtApi._(DynamicLibrary.open(name), name);
    } on ArgumentError catch (e) {
      throw StateError(
        '$name not found. Run `py -I tools/fetch_litert_sdk.py` and rebuild. '
        'Searched: ${searchDirs().join(', ')}. ($e)',
      );
    }
  }

  /// Directory containing the loaded library, or null when loaded by name.
  String? get libraryDir =>
      p.isAbsolute(libraryPath) ? p.dirname(libraryPath) : null;

  // The lookups below are written out (rather than generated) so every
  // signature is checked against the headers in one place.

  late final getStatusString = library
      .lookupFunction<
        Pointer<Utf8> Function(Int32),
        Pointer<Utf8> Function(int)
      >('LiteRtGetStatusString');

  late final _getRuntimeAbiVersion = library
      .lookupFunction<
        Int32 Function(Pointer<Pointer<Utf8>>),
        int Function(Pointer<Pointer<Utf8>>)
      >('LiteRtGetRuntimeAbiVersion');

  /// ABI version string of the loaded `libLiteRt`.
  String get abiVersion {
    final out = calloc<Pointer<Utf8>>();
    try {
      check(_getRuntimeAbiVersion(out), 'GetRuntimeAbiVersion');
      return out.value.toDartString();
    } finally {
      calloc.free(out);
    }
  }

  late final gpuEnvironmentCreate = library
      .lookupFunction<
        Int32 Function(_H, Int32, Pointer<Uint8>),
        int Function(_H, int, Pointer<Uint8>)
      >('LiteRtGpuEnvironmentCreate');
  late final environmentSupportsFp16 = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Bool>),
        int Function(_H, Pointer<Bool>)
      >('LiteRtEnvironmentSupportsFP16');

  late final createEnvironment = library
      .lookupFunction<
        Int32 Function(Int32, Pointer<Uint8>, _HH),
        int Function(int, Pointer<Uint8>, _HH)
      >('LiteRtCreateEnvironment');
  late final destroyEnvironment = library
      .lookupFunction<Void Function(_H), void Function(_H)>(
        'LiteRtDestroyEnvironment',
      );

  late final createModelFromFile = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Utf8>, _HH),
        int Function(_H, Pointer<Utf8>, _HH)
      >('LiteRtCreateModelFromFile');
  late final destroyModel = library
      .lookupFunction<Void Function(_H), void Function(_H)>(
        'LiteRtDestroyModel',
      );
  late final getNumModelSignatures = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Size>),
        int Function(_H, Pointer<Size>)
      >('LiteRtGetNumModelSignatures');
  late final getModelSignature = library
      .lookupFunction<
        Int32 Function(_H, Size, _HH),
        int Function(_H, int, _HH)
      >('LiteRtGetModelSignature');
  late final getSignatureKey = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Pointer<Utf8>>),
        int Function(_H, Pointer<Pointer<Utf8>>)
      >('LiteRtGetSignatureKey');
  late final getNumSignatureInputs = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Size>),
        int Function(_H, Pointer<Size>)
      >('LiteRtGetNumSignatureInputs');
  late final getNumSignatureOutputs = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Size>),
        int Function(_H, Pointer<Size>)
      >('LiteRtGetNumSignatureOutputs');
  late final getSignatureInputName = library
      .lookupFunction<
        Int32 Function(_H, Size, Pointer<Pointer<Utf8>>),
        int Function(_H, int, Pointer<Pointer<Utf8>>)
      >('LiteRtGetSignatureInputName');
  late final getSignatureOutputName = library
      .lookupFunction<
        Int32 Function(_H, Size, Pointer<Pointer<Utf8>>),
        int Function(_H, int, Pointer<Pointer<Utf8>>)
      >('LiteRtGetSignatureOutputName');
  late final getSignatureInputTensorByIndex = library
      .lookupFunction<
        Int32 Function(_H, Size, _HH),
        int Function(_H, int, _HH)
      >('LiteRtGetSignatureInputTensorByIndex');
  late final getSignatureOutputTensorByIndex = library
      .lookupFunction<
        Int32 Function(_H, Size, _HH),
        int Function(_H, int, _HH)
      >('LiteRtGetSignatureOutputTensorByIndex');
  late final getRankedTensorType = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Uint8>),
        int Function(_H, Pointer<Uint8>)
      >('LiteRtGetRankedTensorType');

  late final createOptions = library
      .lookupFunction<Int32 Function(_HH), int Function(_HH)>(
        'LiteRtCreateOptions',
      );
  late final destroyOptions = library
      .lookupFunction<Void Function(_H), void Function(_H)>(
        'LiteRtDestroyOptions',
      );
  late final setOptionsHardwareAccelerators = library
      .lookupFunction<Int32 Function(_H, Int32), int Function(_H, int)>(
        'LiteRtSetOptionsHardwareAccelerators',
      );
  late final createOpaqueOptions = library
      .lookupFunction<
        Int32 Function(Pointer<Utf8>, _H, _H, _HH),
        int Function(Pointer<Utf8>, _H, _H, _HH)
      >('LiteRtCreateOpaqueOptions');
  late final addOpaqueOptions = library
      .lookupFunction<Int32 Function(_H, _H), int Function(_H, _H)>(
        'LiteRtAddOpaqueOptions',
      );

  late final createCompiledModel = library
      .lookupFunction<
        Int32 Function(_H, _H, _H, _HH),
        int Function(_H, _H, _H, _HH)
      >('LiteRtCreateCompiledModel');
  late final destroyCompiledModel = library
      .lookupFunction<Void Function(_H), void Function(_H)>(
        'LiteRtDestroyCompiledModel',
      );
  late final getInputBufferRequirements = library
      .lookupFunction<
        Int32 Function(_H, Size, Size, _HH),
        int Function(_H, int, int, _HH)
      >('LiteRtGetCompiledModelInputBufferRequirements');
  late final getOutputBufferRequirements = library
      .lookupFunction<
        Int32 Function(_H, Size, Size, _HH),
        int Function(_H, int, int, _HH)
      >('LiteRtGetCompiledModelOutputBufferRequirements');
  late final runCompiledModel = library
      .lookupFunction<
        Int32 Function(_H, Size, Size, _HH, Size, _HH),
        int Function(_H, int, int, _HH, int, _HH)
      >('LiteRtRunCompiledModel');
  late final isFullyAccelerated = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Bool>),
        int Function(_H, Pointer<Bool>)
      >('LiteRtCompiledModelIsFullyAccelerated');
  late final getDelegationMetrics = library
      .lookupFunction<Int32 Function(_H, _HH), int Function(_H, _HH)>(
        'LiteRtCompiledModelGetDelegationMetrics',
      );
  late final getNumDelegationAccelerators = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Size>),
        int Function(_H, Pointer<Size>)
      >('LiteRtGetNumDelegationMetricsAccelerators');
  late final getDelegationAccelerator = library
      .lookupFunction<
        Int32 Function(_H, Size, _HH),
        int Function(_H, int, _HH)
      >('LiteRtGetDelegationMetricsAccelerator');
  late final getAcceleratorMetricsName = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Pointer<Utf8>>),
        int Function(_H, Pointer<Pointer<Utf8>>)
      >('LiteRtGetAcceleratorDelegationMetricsName');
  late final getAcceleratorDelegatedNodes = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Int32>),
        int Function(_H, Pointer<Int32>)
      >('LiteRtGetAcceleratorDelegationMetricsDelegatedNodeCount');
  late final getDelegationTotalNodes = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Int32>),
        int Function(_H, Pointer<Int32>)
      >('LiteRtGetDelegationMetricsTotalNodeCount');

  late final getRequirementsNumTypes = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Int32>),
        int Function(_H, Pointer<Int32>)
      >('LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes');
  late final getRequirementsType = library
      .lookupFunction<
        Int32 Function(_H, Int32, Pointer<Int32>),
        int Function(_H, int, Pointer<Int32>)
      >('LiteRtGetTensorBufferRequirementsSupportedTensorBufferType');
  late final getRequirementsSize = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Size>),
        int Function(_H, Pointer<Size>)
      >('LiteRtGetTensorBufferRequirementsBufferSize');
  late final getRequirementsAlignment = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Size>),
        int Function(_H, Pointer<Size>)
      >('LiteRtGetTensorBufferRequirementsAlignment');

  late final createHostTensorBuffer = library
      .lookupFunction<
        Int32 Function(Pointer<Uint8>, Pointer<Uint8>, Size, _H, _HH),
        int Function(Pointer<Uint8>, Pointer<Uint8>, int, _H, _HH)
      >('LiteRtCreateTensorBufferFromHostMemory');
  late final createManagedFromRequirements = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Uint8>, _H, _HH),
        int Function(_H, Pointer<Uint8>, _H, _HH)
      >('LiteRtCreateManagedTensorBufferFromRequirements');
  late final lockTensorBuffer = library
      .lookupFunction<
        Int32 Function(_H, Pointer<Pointer<Void>>, Int32),
        int Function(_H, Pointer<Pointer<Void>>, int)
      >('LiteRtLockTensorBuffer');
  late final unlockTensorBuffer = library
      .lookupFunction<Int32 Function(_H), int Function(_H)>(
        'LiteRtUnlockTensorBuffer',
      );
  late final destroyTensorBuffer = library
      .lookupFunction<Void Function(_H), void Function(_H)>(
        'LiteRtDestroyTensorBuffer',
      );

  /// Throws [LiteRtException] unless [status] is `kLiteRtStatusOk`.
  void check(int status, String call) {
    if (status == 0) return;
    String name;
    try {
      name = getStatusString(status).toDartString();
    } catch (_) {
      name = 'status $status';
    }
    throw LiteRtException(call, status, name);
  }
}
