import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'custom_ops.dart';
import 'ffi.dart';

/// Where a [CompiledGraph] should run, independent of app-level settings.
class GraphConfig {
  const GraphConfig({
    this.gpu = false,
    this.npu = false,
    this.cpuFallback = true,
    this.fp16 = false,
    this.cpuThreads = 4,
  });

  final bool gpu;
  final bool npu;

  /// With [gpu]/[npu]: let LiteRT run unsupported ops on the CPU.
  final bool cpuFallback;

  /// GPU only: allow FP16 math (default FP32).
  final bool fp16;
  final int cpuThreads;

  GraphConfig copyWith({bool? fp16}) => GraphConfig(
    gpu: gpu,
    npu: npu,
    cpuFallback: cpuFallback,
    fp16: fp16 ?? this.fp16,
    cpuThreads: cpuThreads,
  );

  int get hardwareMask {
    var m = 0;
    if (gpu) m |= kHwGpu;
    if (npu) m |= kHwNpu;
    if (m == 0 || cpuFallback) m |= kHwCpu;
    return m;
  }

  List<String> get requested => [
    if (gpu) 'GPU',
    if (npu) 'NPU',
    if (!gpu && !npu || cpuFallback) 'CPU',
  ];
}

/// Dtype of a graph tensor.
enum GraphDType {
  float32(kElementFloat32, 4),
  int32(kElementInt32, 4),
  int64(kElementInt64, 8),
  int16(kElementInt16, 2),
  int8(kElementInt8, 1),
  uint8(kElementUInt8, 1),
  bool_(kElementBool, 1),
  float16(kElementFloat16, 2);

  const GraphDType(this.code, this.bytes);
  final int code;
  final int bytes;

  static GraphDType fromCode(int code) => values.firstWhere(
    (t) => t.code == code,
    orElse: () =>
        throw UnsupportedError('Unsupported LiteRT element type $code'),
  );
}

/// How much of a compiled graph a hardware accelerator took.
class DelegationInfo {
  const DelegationInfo(this.accelerator, this.delegatedNodes, this.totalNodes);
  final String accelerator;
  final int delegatedNodes;
  final int totalNodes;

  /// Readable accelerator name: runtime names like `CpuAccelerator` and
  /// `GPU WebGPU` become `CPU (XNNPACK)` and `GPU (WebGPU)`.
  String get label {
    if (accelerator.startsWith('Cpu')) return 'CPU (XNNPACK)';
    final gpu = RegExp(r'^GPU (\w+)$').firstMatch(accelerator);
    return gpu != null ? 'GPU (${gpu[1]})' : accelerator;
  }

  @override
  String toString() => '$label $delegatedNodes/$totalNodes ops';
}

/// One `.tflite` file compiled with the LiteRT `CompiledModel` C API.
///
/// Every signature is addressable by key, so multi-signature graphs (named
/// `prefill`/`decode` entry points, int32/int64 inputs, dynamic KV caches)
/// work the same as single-signature ones. Lives in one isolate.
class CompiledGraph {
  CompiledGraph._(
    this._api,
    this._env,
    this._model,
    this._options,
    this._compiled,
    this._signatures,
    this.config,
    this.note,
  );

  final LiteRtApi _api;
  final Handle _env;
  final Handle _model;
  final Handle _options;
  final Handle _compiled;
  final List<_SignatureInfo> _signatures;
  /// The config actually used (FP16 may have been downgraded).
  final GraphConfig config;

  /// Set when the requested config was adjusted, e.g. FP16 → FP32.
  final String? note;
  final _open = <String, GraphSignature>{};
  var _closed = false;

  /// Environment shared by every graph in this isolate.
  static Handle? _sharedEnv;

  static Handle _environment(LiteRtApi api) {
    final existing = _sharedEnv;
    if (existing != null) return existing;
    final dir = api.libraryDir;
    final opts = calloc<Uint8>(envOptionBytes);
    Pointer<Utf8>? dirStr;
    try {
      var count = 0;
      if (dir != null) {
        // Points the runtime at the folder holding the GPU accelerator
        // library (libLiteRtWebGpuAccelerator, dxcompiler, ...).
        dirStr = dir.toNativeUtf8();
        opts.cast<Int32>().value = kEnvRuntimeLibraryDir;
        (opts + 8).cast<Int32>().value = kAnyString;
        (opts + 16).cast<Pointer<Utf8>>().value = dirStr;
        count = 1;
      }
      final out = calloc<Pointer<Void>>();
      try {
        api.check(api.createEnvironment(count, opts, out), 'CreateEnvironment');
        return _sharedEnv = out.value;
      } finally {
        calloc.free(out);
      }
    } finally {
      calloc.free(opts);
      if (dirStr != null) malloc.free(dirStr);
    }
  }

  static Handle _opaque(LiteRtApi api, String identifier, String toml) {
    final id = identifier.toNativeUtf8();
    final payload = toml.toNativeUtf8(); // freed by the runtime on destroy
    final out = calloc<Pointer<Void>>();
    try {
      api.check(
        api.createOpaqueOptions(
          id,
          payload.cast(),
          malloc.nativeFree.cast(),
          out,
        ),
        'CreateOpaqueOptions($identifier)',
      );
      return out.value;
    } finally {
      malloc.free(id);
      calloc.free(out);
    }
  }

  /// Whether the GPU adapter can run FP16 shaders; null if it can't be
  /// determined. Without it (e.g. GTX 10-series) the WebGPU kernels fail WGSL
  /// validation and the runtime silently returns garbage.
  ///
  /// Only meaningful once a GPU graph has been compiled (that creates the
  /// runtime's GPU environment).
  static bool? _gpuFp16(LiteRtApi api, Handle env) {
    final b = calloc<Bool>();
    try {
      return api.environmentSupportsFp16(env, b) == 0 ? b.value : null;
    } finally {
      calloc.free(b);
    }
  }

  /// Compiles [path] for [config]. Throws [LiteRtException] if the model
  /// cannot be loaded or compiled for the requested hardware.
  ///
  /// A GPU FP16 request on an adapter without FP16 support is recompiled in
  /// FP32 (see [note]).
  static CompiledGraph open(String path, GraphConfig config) {
    final graph = _compile(path, config, null);
    if (config.gpu &&
        config.fp16 &&
        _gpuFp16(graph._api, graph._env) == false) {
      graph.close();
      return _compile(
        path,
        config.copyWith(fp16: false),
        'This GPU lacks FP16 shader support; compiled in FP32 instead.',
      );
    }
    return graph;
  }

  static CompiledGraph _compile(String path, GraphConfig config, String? note) {
    final api = LiteRtApi.instance;
    final env = _environment(api);
    final out = calloc<Pointer<Void>>();
    final pathStr = path.toNativeUtf8();
    Handle? model, options, compiled;
    try {
      api.check(
        api.createModelFromFile(env, pathStr, out),
        'CreateModelFromFile',
      );
      model = out.value;

      api.check(api.createOptions(out), 'CreateOptions');
      options = out.value;
      registerCustomOps(api, options);
      api.check(
        api.setOptionsHardwareAccelerators(options, config.hardwareMask),
        'SetOptionsHardwareAccelerators',
      );
      api.check(
        api.addOpaqueOptions(
          options,
          _opaque(api, 'cpu_options', 'num_threads = ${config.cpuThreads}\n'),
        ),
        'AddOpaqueOptions(cpu)',
      );
      if (config.gpu) {
        // LiteRtDelegatePrecision: 1 = FP16, 2 = FP32.
        api.check(
          api.addOpaqueOptions(
            options,
            _opaque(api, 'gpu_options', 'precision = ${config.fp16 ? 1 : 2}\n'),
          ),
          'AddOpaqueOptions(gpu)',
        );
      }

      api.check(
        api.createCompiledModel(env, model, options, out),
        'CreateCompiledModel',
      );
      compiled = out.value;
      return CompiledGraph._(
        api,
        env,
        model,
        options,
        compiled,
        _readSignatures(api, model),
        config,
        note,
      );
    } catch (_) {
      if (compiled != null) api.destroyCompiledModel(compiled);
      if (options != null) api.destroyOptions(options);
      if (model != null) api.destroyModel(model);
      rethrow;
    } finally {
      calloc.free(out);
      malloc.free(pathStr);
    }
  }

  static List<_SignatureInfo> _readSignatures(LiteRtApi api, Handle model) {
    final size = calloc<Size>();
    final str = calloc<Pointer<Utf8>>();
    final h = calloc<Pointer<Void>>();
    final type = calloc<Uint8>(rankedTensorTypeBytes);
    try {
      api.check(
        api.getNumModelSignatures(model, size),
        'GetNumModelSignatures',
      );
      final n = size.value;
      final result = <_SignatureInfo>[];
      for (var s = 0; s < n; s++) {
        api.check(api.getModelSignature(model, s, h), 'GetModelSignature');
        final sig = h.value;
        api.check(api.getSignatureKey(sig, str), 'GetSignatureKey');
        final key = str.value.toDartString();

        List<TensorSpec> tensors(bool input) {
          api.check(
            input
                ? api.getNumSignatureInputs(sig, size)
                : api.getNumSignatureOutputs(sig, size),
            'GetNumSignatureTensors',
          );
          final count = size.value;
          return [
            for (var i = 0; i < count; i++)
              () {
                api.check(
                  input
                      ? api.getSignatureInputName(sig, i, str)
                      : api.getSignatureOutputName(sig, i, str),
                  'GetSignatureTensorName',
                );
                final name = str.value.toDartString();
                api.check(
                  input
                      ? api.getSignatureInputTensorByIndex(sig, i, h)
                      : api.getSignatureOutputTensorByIndex(sig, i, h),
                  'GetSignatureTensor',
                );
                api.check(
                  api.getRankedTensorType(h.value, type),
                  'GetRankedTensorType($name)',
                );
                return TensorSpec(
                  name,
                  GraphDType.fromCode(rankedElementType(type)),
                  rankedShape(type),
                );
              }(),
          ];
        }

        result.add(_SignatureInfo(key, s, tensors(true), tensors(false)));
      }
      return result;
    } finally {
      calloc.free(size);
      calloc.free(str);
      calloc.free(h);
      calloc.free(type);
    }
  }

  List<String> get signatureKeys => [for (final s in _signatures) s.key];

  /// Signature [key] (defaults to the first), with allocated I/O buffers.
  GraphSignature signature([String? key]) {
    if (_closed) throw StateError('Graph is closed');
    final info = key == null
        ? _signatures.first
        : _signatures.firstWhere(
            (s) => s.key == key,
            orElse: () => throw ArgumentError(
              'No signature "$key"; have ${signatureKeys.join(', ')}',
            ),
          );
    return _open[info.key] ??= GraphSignature._(this, info);
  }

  /// True when every op ran on a non-CPU accelerator; null if unknown.
  bool? get isFullyAccelerated {
    final b = calloc<Bool>();
    try {
      final st = _api.isFullyAccelerated(_compiled, b);
      return st == 0 ? b.value : null;
    } finally {
      calloc.free(b);
    }
  }

  /// Per-accelerator delegation summary (empty if all ops ran on built-in
  /// CPU kernels or the runtime does not report it).
  List<DelegationInfo> get delegation {
    final h = calloc<Pointer<Void>>();
    final size = calloc<Size>();
    final total = calloc<Int32>();
    final nodes = calloc<Int32>();
    final str = calloc<Pointer<Utf8>>();
    try {
      if (_api.getDelegationMetrics(_compiled, h) != 0) return const [];
      final metrics = h.value;
      if (_api.getDelegationTotalNodes(metrics, total) != 0) total.value = 0;
      if (_api.getNumDelegationAccelerators(metrics, size) != 0) {
        return const [];
      }
      final n = size.value;
      final result = <DelegationInfo>[];
      for (var i = 0; i < n; i++) {
        if (_api.getDelegationAccelerator(metrics, i, h) != 0) continue;
        final acc = h.value;
        if (_api.getAcceleratorMetricsName(acc, str) != 0) continue;
        if (_api.getAcceleratorDelegatedNodes(acc, nodes) != 0) nodes.value = 0;
        result.add(
          DelegationInfo(str.value.toDartString(), nodes.value, total.value),
        );
      }
      return result;
    } finally {
      calloc.free(h);
      calloc.free(size);
      calloc.free(total);
      calloc.free(nodes);
      calloc.free(str);
    }
  }

  /// Short label of where the graph actually runs, e.g. `GPU (WebGPU) 210/214`.
  String get acceleratorLabel {
    final d = delegation.where((e) => e.delegatedNodes > 0).toList();
    if (d.isEmpty) return 'CPU';
    return d.join(' + ');
  }

  void close() {
    if (_closed) return;
    _closed = true;
    for (final s in _open.values) {
      s._dispose();
    }
    _open.clear();
    _api.destroyCompiledModel(_compiled);
    _api.destroyOptions(_options);
    _api.destroyModel(_model);
  }
}

class _SignatureInfo {
  _SignatureInfo(this.key, this.index, this.inputs, this.outputs);
  final String key;
  final int index;
  final List<TensorSpec> inputs;
  final List<TensorSpec> outputs;
}

/// Static description of a signature tensor.
class TensorSpec {
  TensorSpec(this.name, this.dtype, this.shape);
  final String name;
  final GraphDType dtype;
  final List<int> shape;

  int get elements => shape.fold(1, (a, b) => a * b);
  int get bytes => elements * dtype.bytes;

  @override
  String toString() => '$name ${dtype.name}$shape';
}

/// Host-visible tensor. Reads/writes go through [data]; for accelerators
/// that cannot use host memory directly the signature copies in/out around
/// each [GraphSignature.invoke].
class GraphTensor {
  GraphTensor._(this.spec, this._raw, this.data);

  final TensorSpec spec;
  final Pointer<Uint8> _raw;

  /// Aligned, zero-initialised host memory of [numBytes].
  final Pointer<Uint8> data;

  Handle? _buffer; // LiteRtTensorBuffer
  var _managed = false; // true: staged via lock/unlock

  List<int> get shape => spec.shape;
  GraphDType get dtype => spec.dtype;

  /// True when the accelerator can't read host memory directly, so the tensor
  /// is staged through a LiteRT-managed buffer around each invoke.
  bool get isStaged => _managed;
  int numBytes() => spec.bytes;

  Float32List? _f32;
  Int32List? _i32;
  Int64List? _i64;

  Float32List asFloat32View() {
    _expect(GraphDType.float32);
    return _f32 ??= data.cast<Float>().asTypedList(spec.elements);
  }

  Int32List asInt32View() {
    _expect(GraphDType.int32);
    return _i32 ??= data.cast<Int32>().asTypedList(spec.elements);
  }

  Int64List asInt64View() {
    _expect(GraphDType.int64);
    return _i64 ??= data.cast<Int64>().asTypedList(spec.elements);
  }

  void _expect(GraphDType t) {
    if (spec.dtype != t) {
      throw StateError(
        'Tensor ${spec.name} is ${spec.dtype.name}, not ${t.name}',
      );
    }
  }

  /// Copies [src] (raw bytes of any typed list) into the tensor.
  void setTo(TypedData src) {
    if (src.lengthInBytes > spec.bytes) {
      throw ArgumentError(
        '${src.lengthInBytes} bytes do not fit ${spec.name} (${spec.bytes} B)',
      );
    }
    data
        .asTypedList(spec.bytes)
        .setAll(
          0,
          src.buffer.asUint8List(src.offsetInBytes, src.lengthInBytes),
        );
  }
}

/// A signature with allocated tensors. Views from [inF]/[outF] alias native
/// memory; read outputs after [invoke].
class GraphSignature {
  GraphSignature._(this._graph, _SignatureInfo info)
    : key = info.key,
      _index = info.index {
    try {
      for (var i = 0; i < info.inputs.length; i++) {
        _inputs.add(_allocate(info.inputs[i], i, input: true));
      }
      for (var i = 0; i < info.outputs.length; i++) {
        _outputs.add(_allocate(info.outputs[i], i, input: false));
      }
    } catch (_) {
      _dispose();
      rethrow;
    }
    _inArr = calloc<Pointer<Void>>(_inputs.length);
    _outArr = calloc<Pointer<Void>>(_outputs.length);
    for (var i = 0; i < _inputs.length; i++) {
      _inArr![i] = _inputs[i]._buffer!;
    }
    for (var i = 0; i < _outputs.length; i++) {
      _outArr![i] = _outputs[i]._buffer!;
    }
    // Tensor memory is calloc'd: some graphs misbehave on uninitialised
    // (NaN) inputs, e.g. during warm-up, so everything starts zeroed.
  }

  final CompiledGraph _graph;
  final String key;
  final int _index;
  final _inputs = <GraphTensor>[];
  final _outputs = <GraphTensor>[];
  Pointer<Pointer<Void>>? _inArr, _outArr;
  final _lockPtr = calloc<Pointer<Void>>();
  var _disposed = false;

  List<String> get inputNames => [for (final t in _inputs) t.spec.name];
  List<String> get outputNames => [for (final t in _outputs) t.spec.name];

  GraphTensor input(String name) => _inputs.firstWhere(
    (t) => t.spec.name == name,
    orElse: () => throw ArgumentError('No input "$name" in signature $key'),
  );
  GraphTensor output(String name) => _outputs.firstWhere(
    (t) => t.spec.name == name,
    orElse: () => throw ArgumentError('No output "$name" in signature $key'),
  );

  GraphTensor inputAt(int i) => _inputs[i];
  GraphTensor outputAt(int i) => _outputs[i];

  Float32List inF(String name) => input(name).asFloat32View();
  Float32List outF(String name) => output(name).asFloat32View();

  void setInt32(String name, List<int> values) =>
      input(name).setTo(Int32List.fromList(values));

  void setInt64(String name, List<int> values) =>
      input(name).setTo(Int64List.fromList(values));

  GraphTensor _allocate(TensorSpec spec, int i, {required bool input}) {
    final api = _graph._api;
    if (spec.shape.any((d) => d <= 0)) {
      throw UnsupportedError(
        'Dynamic shape ${spec.shape} for ${spec.name} in signature $key',
      );
    }
    final req = calloc<Pointer<Void>>();
    final sizeP = calloc<Size>();
    final countP = calloc<Int32>();
    final typeP = calloc<Int32>();
    final buf = calloc<Pointer<Void>>();
    final type = calloc<Uint8>(rankedTensorTypeBytes);
    try {
      // Rebuild the LiteRtRankedTensorType: element type + layout.
      type.cast<Int32>().value = spec.dtype.code;
      final layout = type + 4;
      layout.value = spec.shape.length & 0x7f;
      final dims = (layout + 4).cast<Int32>();
      for (var d = 0; d < spec.shape.length; d++) {
        dims[d] = spec.shape[d];
      }

      api.check(
        input
            ? api.getInputBufferRequirements(_graph._compiled, _index, i, req)
            : api.getOutputBufferRequirements(_graph._compiled, _index, i, req),
        'GetBufferRequirements(${spec.name})',
      );
      final reqH = req.value;
      api.check(api.getRequirementsSize(reqH, sizeP), 'GetRequirementsSize');
      final reqBytes = sizeP.value;
      var align = 64;
      if (api.getRequirementsAlignment(reqH, sizeP) == 0 &&
          sizeP.value > align) {
        align = sizeP.value;
      }
      api.check(
        api.getRequirementsNumTypes(reqH, countP),
        'GetRequirementsNumTypes',
      );
      var hostOk = false;
      for (var t = 0; t < countP.value; t++) {
        api.check(
          api.getRequirementsType(reqH, t, typeP),
          'GetRequirementsType',
        );
        if (typeP.value == kBufferHostMemory) hostOk = true;
      }

      final bytes = spec.bytes;
      final alloc = (reqBytes > bytes ? reqBytes : bytes) + align;
      final raw = calloc<Uint8>(alloc);
      final aligned = Pointer<Uint8>.fromAddress(
        (raw.address + align - 1) & ~(align - 1),
      );
      final tensor = GraphTensor._(spec, raw, aligned);
      try {
        if (hostOk) {
          api.check(
            api.createHostTensorBuffer(
              type,
              aligned,
              alloc - (aligned.address - raw.address),
              nullptr,
              buf,
            ),
            'CreateTensorBufferFromHostMemory(${spec.name})',
          );
        } else {
          api.check(
            api.createManagedFromRequirements(_graph._env, type, reqH, buf),
            'CreateManagedTensorBuffer(${spec.name})',
          );
          tensor._managed = true;
        }
      } catch (_) {
        calloc.free(raw);
        rethrow;
      }
      tensor._buffer = buf.value;
      return tensor;
    } finally {
      calloc.free(req);
      calloc.free(sizeP);
      calloc.free(countP);
      calloc.free(typeP);
      calloc.free(buf);
      calloc.free(type);
    }
  }

  /// Runs the signature on the inputs currently written to the tensors.
  void invoke() {
    if (_disposed) throw StateError('Signature is disposed');
    final api = _graph._api;
    for (final t in _inputs) {
      if (t._managed) _copy(api, t, toDevice: true);
    }
    api.check(
      api.runCompiledModel(
        _graph._compiled,
        _index,
        _inputs.length,
        _inArr!,
        _outputs.length,
        _outArr!,
      ),
      'RunCompiledModel($key)',
    );
    for (final t in _outputs) {
      if (t._managed) _copy(api, t, toDevice: false);
    }
  }

  void _copy(LiteRtApi api, GraphTensor t, {required bool toDevice}) {
    api.check(
      api.lockTensorBuffer(
        t._buffer!,
        _lockPtr,
        toDevice ? kLockWrite : kLockRead,
      ),
      'LockTensorBuffer(${t.spec.name})',
    );
    try {
      final n = t.spec.bytes;
      final device = _lockPtr.value.cast<Uint8>().asTypedList(n);
      final host = t.data.asTypedList(n);
      if (toDevice) {
        device.setAll(0, host);
      } else {
        host.setAll(0, device);
      }
    } finally {
      api.unlockTensorBuffer(t._buffer!);
    }
  }

  void _dispose() {
    if (_disposed) return;
    _disposed = true;
    final api = _graph._api;
    for (final t in [..._inputs, ..._outputs]) {
      final b = t._buffer;
      if (b != null) api.destroyTensorBuffer(b);
      calloc.free(t._raw);
    }
    _inputs.clear();
    _outputs.clear();
    calloc.free(_lockPtr);
    if (_inArr != null) calloc.free(_inArr!);
    if (_outArr != null) calloc.free(_outArr!);
  }
}
