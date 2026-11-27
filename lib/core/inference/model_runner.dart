import 'dart:typed_data';

import '../litert/compiled_graph.dart';
import '../models/model_spec.dart';
import 'inference_options.dart';

/// Maps app-level [InferenceOptions] to a LiteRT [GraphConfig].
GraphConfig graphConfigFor(InferenceOptions o, {int? threads}) => GraphConfig(
  gpu: o.accelerator == AcceleratorKind.gpu,
  npu: o.accelerator == AcceleratorKind.npu,
  cpuFallback: o.cpuFallback,
  fp16: o.precision == PrecisionKind.fp16,
  cpuThreads: threads ?? o.cpuThreads,
);

/// A loaded single-input / single-output float model, run through the LiteRT
/// `CompiledModel` C API.
///
/// Lives inside the inference worker isolate; never crosses isolates.
class ModelRunner {
  ModelRunner._(this._graph, this._sig, this._options);

  final CompiledGraph _graph;
  final GraphSignature _sig;
  final InferenceOptions _options;

  static ModelRunner load(String path, ModelSpec spec, InferenceOptions o) {
    final graph = CompiledGraph.open(path, graphConfigFor(o));
    try {
      final sig = graph.signature();
      final in0 = sig.inputAt(0), out0 = sig.outputAt(0);
      if (sig.inputNames.length != 1 ||
          sig.outputNames.length != 1 ||
          in0.dtype != GraphDType.float32 ||
          out0.dtype != GraphDType.float32 ||
          in0.numBytes() != spec.inputElements * 4 ||
          out0.numBytes() != spec.outputElements * 4) {
        throw StateError(
          'Model I/O does not match spec ${spec.inputShape} → '
          '${spec.outputShape} (float32): '
          'in=${in0.spec} out=${out0.spec}',
        );
      }
      return ModelRunner._(graph, sig, o);
    } catch (_) {
      graph.close();
      rethrow;
    }
  }

  /// Runs one inference. The returned list aliases native memory and is
  /// overwritten by the next call.
  Float32List run(Float32List input) {
    _sig.inputAt(0).asFloat32View().setAll(0, input);
    _sig.invoke();
    return _sig.outputAt(0).asFloat32View();
  }

  void close() => _graph.close();

  String get runtimeLabel => 'LiteRT CompiledModel';

  List<String> get requestedAccelerators => graphConfigFor(_options).requested;

  /// What actually ran the graph, e.g. `GPU (WebGPU) 210/214 ops`.
  List<String> get actualAccelerators {
    final d = _graph.delegation.where((e) => e.delegatedNodes > 0).toList();
    return d.isEmpty ? const ['CPU'] : [for (final e in d) e.label];
  }

  bool get didFallback {
    final wantsHw = _options.accelerator != AcceleratorKind.cpu;
    return wantsHw && _graph.isFullyAccelerated != true;
  }

  bool? get fullyAccelerated => _graph.isFullyAccelerated;

  String? get fallbackReason {
    final parts = [
      if (_graph.note != null) _graph.note!,
      if (didFallback)
        'Not every op ran on ${_options.accelerator.label}: '
            '${_graph.delegation.join(', ')}; the rest used CPU kernels.',
    ];
    return parts.isEmpty ? null : parts.join(' ');
  }
}
