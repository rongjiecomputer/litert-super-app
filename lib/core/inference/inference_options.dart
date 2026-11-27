import 'dart:io';
import 'dart:math' as math;

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

/// LiteRT `CompiledModel` inference configuration, persisted per use case.
class InferenceOptions {
  const InferenceOptions({
    this.accelerator = AcceleratorKind.gpu,
    this.precision = PrecisionKind.fp32,
    this.cpuFallback = true,
    this.cpuThreads = 4,
  });

  final AcceleratorKind accelerator;
  final PrecisionKind precision;

  /// Let LiteRT run ops the GPU/NPU can't handle on the CPU.
  final bool cpuFallback;

  /// XNNPACK thread count (CPU compute, and CPU fallback ops).
  final int cpuThreads;

  InferenceOptions copyWith({
    AcceleratorKind? accelerator,
    PrecisionKind? precision,
    bool? cpuFallback,
    int? cpuThreads,
  }) => InferenceOptions(
    accelerator: accelerator ?? this.accelerator,
    precision: precision ?? this.precision,
    cpuFallback: cpuFallback ?? this.cpuFallback,
    cpuThreads: cpuThreads ?? this.cpuThreads,
  );

  /// Returns options valid for [caps], replacing unsupported choices.
  InferenceOptions sanitized(PlatformCaps caps) {
    final accel = caps.accelerators.contains(accelerator)
        ? accelerator
        : caps.accelerators.first;
    var o = copyWith(
      accelerator: accel,
      cpuThreads: cpuThreads.clamp(1, caps.maxThreads),
    );
    if (!caps.supportsPrecision(o)) {
      o = o.copyWith(precision: PrecisionKind.fp32);
    }
    return o;
  }

  /// Short human description, e.g. "GPU · FP32" or "CPU · 4T".
  String get summary => accelerator == AcceleratorKind.cpu
      ? '${accelerator.label} · ${cpuThreads}T'
      : '${accelerator.label} · ${precision.label}';

  Map<String, Object?> toJson() => {
    'accelerator': accelerator.name,
    'precision': precision.name,
    'cpuFallback': cpuFallback,
    'cpuThreads': cpuThreads,
  };

  /// Unknown or legacy keys (e.g. the old `runtime` / `bufferMode`) are
  /// ignored.
  factory InferenceOptions.fromJson(Map<String, Object?> json) {
    T pick<T extends Enum>(List<T> values, Object? name, T fallback) =>
        values.where((v) => v.name == name).firstOrNull ?? fallback;
    const d = InferenceOptions();
    return InferenceOptions(
      accelerator: pick(
        AcceleratorKind.values,
        json['accelerator'],
        d.accelerator,
      ),
      precision: pick(PrecisionKind.values, json['precision'], d.precision),
      cpuFallback: json['cpuFallback'] as bool? ?? d.cpuFallback,
      cpuThreads: (json['cpuThreads'] as num?)?.toInt() ?? d.cpuThreads,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is InferenceOptions &&
      other.accelerator == accelerator &&
      other.precision == precision &&
      other.cpuFallback == cpuFallback &&
      other.cpuThreads == cpuThreads;

  @override
  int get hashCode =>
      Object.hash(accelerator, precision, cpuFallback, cpuThreads);
}

/// What the current OS build of the LiteRT runtime can actually run.
class PlatformCaps {
  const PlatformCaps({
    required this.os,
    required this.accelerators,
    required this.maxThreads,
  });

  final String os;
  final List<AcceleratorKind> accelerators;
  final int maxThreads;

  static PlatformCaps current() =>
      forOs(Platform.operatingSystem, math.max(1, Platform.numberOfProcessors));

  static PlatformCaps forOs(String os, int cores) {
    const cpu = AcceleratorKind.cpu, gpu = AcceleratorKind.gpu;
    const npu = AcceleratorKind.npu;
    return switch (os) {
      // Metal GPU + Core ML NPU.
      'macos' || 'ios' => PlatformCaps(
        os: os,
        accelerators: const [cpu, gpu, npu],
        maxThreads: cores,
      ),
      // OpenCL/GL/WebGPU GPU, Qualcomm NPU (needs vendor runtime).
      'android' => PlatformCaps(
        os: os,
        accelerators: const [cpu, gpu, npu],
        maxThreads: cores,
      ),
      // Windows/Linux: WebGPU (D3D12 / Vulkan) GPU. The Windows x64 NPU
      // (Intel OpenVINO) needs a separate runtime, so it isn't offered.
      _ => PlatformCaps(
        os: os,
        accelerators: const [cpu, gpu],
        maxThreads: cores,
      ),
    };
  }

  /// FP16 applies to GPU compilation only.
  bool supportsPrecision(InferenceOptions o) =>
      o.accelerator == AcceleratorKind.gpu;

  bool get supportsVideo => os == 'windows' || os == 'macos' || os == 'linux';
}
