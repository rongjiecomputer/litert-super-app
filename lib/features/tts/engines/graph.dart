import '../../../core/inference/inference_options.dart';
import '../../../core/inference/model_runner.dart';
import '../../../core/litert/compiled_graph.dart';

/// A signature with allocated tensors; see [GraphSignature].
typedef Sig = GraphSignature;

/// One `.tflite` file compiled with the LiteRT `CompiledModel` API,
/// exposing its signatures with zero-copy tensor views.
class LiteRtGraph {
  LiteRtGraph._(this._graph);

  final CompiledGraph _graph;

  /// Where the graph actually runs, e.g. `GPU (WebGPU)` or `CPU`.
  String get accelerator => _graph.acceleratorLabel;

  /// True when every op ran on the requested GPU/NPU.
  bool? get fullyAccelerated => _graph.isFullyAccelerated;

  /// [threads] overrides the CPU thread count (tiny graphs run best on 1).
  ///
  /// With a GPU/NPU selected, only graphs passing [accelerate] use it; the
  /// others run on the CPU. TTS pipelines are many graphs of very different
  /// shape: measured on a GTX 1050, the Qwen3 codec decoder is ~8x faster on
  /// WebGPU, while Kokoro's graphs are slower or fail there (see
  /// `kokoro_engine.dart`).
  static LiteRtGraph open(
    String path,
    InferenceOptions o, {
    int? threads,
    bool accelerate = false,
  }) {
    final effective = accelerate || o.accelerator == AcceleratorKind.cpu
        ? o
        : o.copyWith(accelerator: AcceleratorKind.cpu);
    return LiteRtGraph._(
      CompiledGraph.open(path, graphConfigFor(effective, threads: threads)),
    );
  }

  List<String> get signatureKeys => _graph.signatureKeys;

  /// Signature [key] (defaults to the first, usually `serving_default`).
  Sig sig([String? key]) => _graph.signature(key);

  void close() => _graph.close();
}
