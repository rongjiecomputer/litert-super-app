import 'dart:typed_data';

import 'package:flutter_litert/native.dart';

import '../../../core/inference/inference_options.dart';
import '../../../core/inference/model_runner.dart';

/// One `.tflite` file opened with the classic Interpreter, exposing its
/// signatures with zero-copy float32 tensor views.
class LiteRtGraph {
  LiteRtGraph._(this._bundle);

  final InterpreterBundle _bundle;
  final Map<String, Sig> _sigs = {};

  String get accelerator => _bundle.accelerator;
  bool get delegateActive => _bundle.delegateActive;

  static LiteRtGraph open(String path, InferenceOptions o, {int? threads}) =>
      LiteRtGraph._(InterpreterBundle.open(path, o, threads: threads));

  List<String> get signatureKeys => _bundle.interpreter.signatureKeys;

  /// Signature [key] (defaults to the first one, usually `serving_default`).
  Sig sig([String? key]) {
    final k = key ?? signatureKeys.first;
    return _sigs[k] ??= Sig._(_bundle.interpreter.getSignatureRunner(k));
  }

  void close() {
    for (final s in _sigs.values) {
      s._runner.close();
    }
    _sigs.clear();
    _bundle.close();
  }
}

/// A signature with allocated tensors. Views returned by [inF]/[outF]
/// alias native memory; re-fetch them after each [invoke] for outputs.
class Sig {
  Sig._(this._runner) {
    _runner.allocateTensors();
    // Arena memory is uninitialised; some graphs fail on garbage (NaN)
    // inputs, e.g. during warm-up, so start from zeros.
    for (final n in _runner.inputNames) {
      final t = _runner.getInputTensor(n);
      t.setTo(Uint8List(t.numBytes()));
    }
  }

  final SignatureRunner _runner;

  List<String> get inputNames => _runner.inputNames;
  List<String> get outputNames => _runner.outputNames;

  Tensor input(String name) => _runner.getInputTensor(name);
  Tensor output(String name) => _runner.getOutputTensor(name);

  Float32List inF(String name) => input(name).asFloat32View();
  Float32List outF(String name) => output(name).asFloat32View();

  void setInt32(String name, List<int> values) =>
      input(name).setTo(Int32List.fromList(values));

  void setInt64(String name, List<int> values) =>
      input(name).setTo(Int64List.fromList(values));

  void invoke() => _runner.invoke();
}
