import 'dart:io';
import 'dart:math' as math;

/// Which LiteRT API executes the model.
enum RuntimeKind {
  /// LiteRT 2.x `CompiledModel` (CPU / GPU / NPU accelerators).
  compiledModel('CompiledModel'),

  /// Classic TFLite `Interpreter` with delegates (XNNPACK, Metal).
  interpreter('Interpreter');

  const RuntimeKind(this.label);
  final String label;
}

enum AcceleratorKind {
  cpu('CPU'),
  gpu('GPU'),
  npu('NPU');

  const AcceleratorKind(this.label);
  final String label;
}

enum PrecisionKind {
  fp32('FP32'),
  fp16('FP16');

  const PrecisionKind(this.label);
  final String label;
}

enum BufferModeKind {
  managed('Managed'),
  hostMemory('Host memory');

  const BufferModeKind(this.label);
  final String label;
}

/// Runtime-agnostic inference configuration, persisted per use case.
class InferenceOptions {
  const InferenceOptions({
    this.runtime = RuntimeKind.compiledModel,
    this.accelerator = AcceleratorKind.gpu,
    this.precision = PrecisionKind.fp32,
    this.cpuFallback = true,
    this.bufferMode = BufferModeKind.managed,
    this.cpuThreads = 4,
  });

  final RuntimeKind runtime;
  final AcceleratorKind accelerator;
  final PrecisionKind precision;

  /// CompiledModel only: retry the whole model on CPU if GPU/NPU fails.
  final bool cpuFallback;

  /// CompiledModel only: LiteRT-managed vs package-owned host buffers.
  final BufferModeKind bufferMode;

  /// Interpreter only: XNNPACK / interpreter thread count.
  final int cpuThreads;

  InferenceOptions copyWith({
    RuntimeKind? runtime,
    AcceleratorKind? accelerator,
    PrecisionKind? precision,
    bool? cpuFallback,
    BufferModeKind? bufferMode,
    int? cpuThreads,
  }) => InferenceOptions(
    runtime: runtime ?? this.runtime,
    accelerator: accelerator ?? this.accelerator,
    precision: precision ?? this.precision,
    cpuFallback: cpuFallback ?? this.cpuFallback,
    bufferMode: bufferMode ?? this.bufferMode,
    cpuThreads: cpuThreads ?? this.cpuThreads,
  );

  /// Returns options valid for [caps], replacing unsupported choices.
  InferenceOptions sanitized(PlatformCaps caps) {
    final rt = caps.runtimes.contains(runtime) ? runtime : caps.runtimes.first;
    final accels = caps.accelerators(rt);
    final accel = accels.contains(accelerator) ? accelerator : accels.first;
    var o = copyWith(
      runtime: rt,
      accelerator: accel,
      cpuThreads: cpuThreads.clamp(1, caps.maxThreads),
    );
    if (!caps.supportsPrecision(o)) {
      o = o.copyWith(precision: PrecisionKind.fp32);
    }
    return o;
  }

  /// Short human description, e.g. "CompiledModel · GPU · FP32".
  String get summary {
    final parts = <String>[runtime.label, accelerator.label, precision.label];
    if (runtime == RuntimeKind.interpreter &&
        accelerator == AcceleratorKind.cpu) {
      parts.add('${cpuThreads}T');
    }
    if (runtime == RuntimeKind.compiledModel &&
        bufferMode == BufferModeKind.hostMemory) {
      parts.add('host buf');
    }
    return parts.join(' · ');
  }

  Map<String, Object?> toJson() => {
    'runtime': runtime.name,
    'accelerator': accelerator.name,
    'precision': precision.name,
    'cpuFallback': cpuFallback,
    'bufferMode': bufferMode.name,
    'cpuThreads': cpuThreads,
  };

  factory InferenceOptions.fromJson(Map<String, Object?> json) {
    T pick<T extends Enum>(List<T> values, Object? name, T fallback) =>
        values.where((v) => v.name == name).firstOrNull ?? fallback;
    const d = InferenceOptions();
    return InferenceOptions(
      runtime: pick(RuntimeKind.values, json['runtime'], d.runtime),
      accelerator: pick(
        AcceleratorKind.values,
        json['accelerator'],
        d.accelerator,
      ),
      precision: pick(PrecisionKind.values, json['precision'], d.precision),
      cpuFallback: json['cpuFallback'] as bool? ?? d.cpuFallback,
      bufferMode: pick(BufferModeKind.values, json['bufferMode'], d.bufferMode),
      cpuThreads: (json['cpuThreads'] as num?)?.toInt() ?? d.cpuThreads,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is InferenceOptions &&
      other.runtime == runtime &&
      other.accelerator == accelerator &&
      other.precision == precision &&
      other.cpuFallback == cpuFallback &&
      other.bufferMode == bufferMode &&
      other.cpuThreads == cpuThreads;

  @override
  int get hashCode => Object.hash(
    runtime,
    accelerator,
    precision,
    cpuFallback,
    bufferMode,
    cpuThreads,
  );
}

/// What the current OS build of flutter_litert can actually run.
class PlatformCaps {
  const PlatformCaps({
    required this.os,
    required this.compiledModelAccelerators,
    required this.interpreterAccelerators,
    required this.maxThreads,
    this.runtimes = RuntimeKind.values,
  });

  final String os;

  /// Runtimes a use case may pick (TTS graphs need the Interpreter for
  /// int32 inputs and named signatures).
  final List<RuntimeKind> runtimes;
  final List<AcceleratorKind> compiledModelAccelerators;
  final List<AcceleratorKind> interpreterAccelerators;
  final int maxThreads;

  static PlatformCaps current() =>
      forOs(Platform.operatingSystem, math.max(1, Platform.numberOfProcessors));

  static PlatformCaps forOs(String os, int cores) {
    const cpu = AcceleratorKind.cpu, gpu = AcceleratorKind.gpu;
    const npu = AcceleratorKind.npu;
    return switch (os) {
      // CompiledModel: Metal GPU + Core ML NPU; Interpreter: XNNPACK + Metal.
      'macos' || 'ios' => PlatformCaps(
        os: os,
        compiledModelAccelerators: const [cpu, gpu, npu],
        interpreterAccelerators: const [cpu, gpu],
        maxThreads: cores,
      ),
      // CompiledModel: OpenCL/GL GPU, Qualcomm NPU (needs vendor runtime).
      'android' => PlatformCaps(
        os: os,
        compiledModelAccelerators: const [cpu, gpu, npu],
        interpreterAccelerators: const [cpu],
        maxThreads: cores,
      ),
      // Windows/Linux: WebGPU (D3D12 / Vulkan) GPU, no NPU.
      _ => PlatformCaps(
        os: os,
        compiledModelAccelerators: const [cpu, gpu],
        interpreterAccelerators: const [cpu],
        maxThreads: cores,
      ),
    };
  }

  /// Copy limited to [allowed] runtimes.
  PlatformCaps restrictTo(List<RuntimeKind> allowed) => PlatformCaps(
    os: os,
    compiledModelAccelerators: compiledModelAccelerators,
    interpreterAccelerators: interpreterAccelerators,
    maxThreads: maxThreads,
    runtimes: allowed,
  );

  List<AcceleratorKind> accelerators(RuntimeKind runtime) => switch (runtime) {
    RuntimeKind.compiledModel => compiledModelAccelerators,
    RuntimeKind.interpreter => interpreterAccelerators,
  };

  /// FP16 applies to GPU compilation, and to XNNPACK's force-FP16 flag.
  bool supportsPrecision(InferenceOptions o) => switch (o.accelerator) {
    AcceleratorKind.gpu => true,
    AcceleratorKind.cpu => o.runtime == RuntimeKind.interpreter,
    AcceleratorKind.npu => false,
  };

  bool get supportsVideo => os == 'windows' || os == 'macos' || os == 'linux';
}
