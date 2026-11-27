import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';

import '../models/model_spec.dart';
import 'inference_options.dart';

/// A loaded single-input / single-output float model.
///
/// Lives inside the inference worker isolate; never crosses isolates.
abstract class ModelRunner {
  /// Runs one inference. The returned list may be reused by the next call.
  Float32List run(Float32List input);

  void close();

  String get runtimeLabel;
  List<String> get requestedAccelerators;
  List<String> get actualAccelerators;
  bool get didFallback;
  bool? get fullyAccelerated;
  String? get fallbackReason;

  static ModelRunner load(String path, ModelSpec spec, InferenceOptions o) =>
      switch (o.runtime) {
        RuntimeKind.compiledModel => _CompiledModelRunner.load(path, spec, o),
        RuntimeKind.interpreter => _InterpreterRunner.load(path, spec, o),
      };
}

class _CompiledModelRunner implements ModelRunner {
  _CompiledModelRunner(this._model, this.fallbackReason);

  final CompiledModel _model;
  @override
  final String? fallbackReason;

  static _CompiledModelRunner load(
    String path,
    ModelSpec spec,
    InferenceOptions o,
  ) {
    final policy = switch (o.accelerator) {
      AcceleratorKind.cpu => CompiledModelPolicy.cpu,
      AcceleratorKind.gpu =>
        o.cpuFallback
            ? CompiledModelPolicy.gpuWithCpuFallback
            : CompiledModelPolicy.gpu,
      AcceleratorKind.npu =>
        o.cpuFallback
            ? CompiledModelPolicy.npuWithCpuFallback
            : CompiledModelPolicy.npu,
    };
    String? fallback;
    final model = CompiledModel.fromFileWithConfig(
      path,
      config: CompiledModelConfig(
        policy: policy,
        precision: o.precision == PrecisionKind.fp16
            ? Precision.fp16
            : Precision.fp32,
        tensorBufferMode: o.bufferMode == BufferModeKind.hostMemory
            ? TensorBufferMode.hostMemory
            : TensorBufferMode.managed,
      ),
      onFallback: (e) => fallback = '$e',
    );
    final inBytes = spec.inputElements * 4;
    final outBytes = spec.outputElements * 4;
    if (model.inputCount != 1 ||
        model.outputCount != 1 ||
        model.inputByteSizes.first != inBytes ||
        model.outputByteSizes.first != outBytes) {
      final got =
          'in=${model.inputByteSizes} out=${model.outputByteSizes} bytes';
      model.close();
      throw StateError(
        'Model I/O does not match spec ${spec.inputShape} → '
        '${spec.outputShape} (float32): $got',
      );
    }
    return _CompiledModelRunner(model, fallback);
  }

  @override
  Float32List run(Float32List input) {
    if (_model.tensorBufferMode == TensorBufferMode.hostMemory) {
      _model.writeInput(0, (buf) => buf.setAll(0, input));
      _model.dispatch();
      return _model.readOutput(0, Float32List.fromList);
    }
    return _model.run([input]).first;
  }

  @override
  void close() => _model.close();

  @override
  String get runtimeLabel => 'LiteRT CompiledModel';
  @override
  List<String> get requestedAccelerators =>
      _model.requestedAccelerators.map((a) => a.name.toUpperCase()).toList();
  @override
  List<String> get actualAccelerators =>
      _model.accelerators.map((a) => a.name.toUpperCase()).toList();
  @override
  bool get didFallback => _model.didFallback;
  @override
  bool? get fullyAccelerated => _model.isFullyAccelerated;
}

/// A classic TFLite [Interpreter] plus the options/delegate it owns, built
/// from [InferenceOptions] (XNNPACK on CPU, Metal delegate on Apple GPU).
///
/// Shared by [ModelRunner] and multi-graph engines (e.g. TTS) that drive
/// signatures directly. Tensors are not allocated yet.
class InterpreterBundle {
  InterpreterBundle._(this.interpreter, this.options, this.delegate, this.accelerator);

  final Interpreter interpreter;
  final InterpreterOptions options;
  final Delegate? delegate;

  /// Human label, e.g. "CPU (XNNPACK 4T)".
  final String accelerator;

  bool get delegateActive => interpreter.hasActiveDelegate;

  static InterpreterBundle open(
    String path,
    InferenceOptions o, {
    int? threads,
  }) {
    final t = threads ?? o.cpuThreads;
    final options = InterpreterOptions()..threads = t;
    Delegate? delegate;
    String accel;
    switch (o.accelerator) {
      case AcceleratorKind.gpu when Platform.isMacOS || Platform.isIOS:
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
            allowPrecisionLoss: o.precision == PrecisionKind.fp16,
          ),
        );
        accel = 'GPU (Metal delegate)';
      case AcceleratorKind.gpu:
      case AcceleratorKind.npu:
        options.delete();
        throw UnsupportedError(
          'Interpreter ${o.accelerator.label} is not available on '
          '${Platform.operatingSystem}; use CompiledModel instead.',
        );
      case AcceleratorKind.cpu:
        delegate = XNNPackDelegate(
          options: XNNPackDelegateOptions(
            numThreads: t,
            flags: o.precision == PrecisionKind.fp16
                ? _xnnFlagQs8 | _xnnFlagQu8 | _xnnFlagForceFp16
                : 0,
          ),
        );
        accel = 'CPU (XNNPACK ${t}T)';
    }
    options.addDelegate(delegate);
    try {
      final interpreter = Interpreter.fromFile(File(path), options: options);
      return InterpreterBundle._(interpreter, options, delegate, accel);
    } catch (_) {
      options.delete();
      delegate.delete();
      rethrow;
    }
  }

  void close() {
    interpreter.close();
    options.delete();
    delegate?.delete();
  }
}

// TfLiteXNNPackDelegateOptions.flags (xnnpack_delegate.h); not re-exported
// by flutter_litert.
const _xnnFlagQs8 = 1;
const _xnnFlagQu8 = 2;
const _xnnFlagForceFp16 = 4;

class _InterpreterRunner implements ModelRunner {
  _InterpreterRunner(
    this._interpreter,
    this._options,
    this._delegate,
    this._accel,
    this._out,
  );

  final Interpreter _interpreter;
  final InterpreterOptions _options;
  final Delegate? _delegate;
  final String _accel;
  final Float32List _out;

  static _InterpreterRunner load(
    String path,
    ModelSpec spec,
    InferenceOptions o,
  ) {
    final b = InterpreterBundle.open(path, o);
    final interpreter = b.interpreter, options = b.options;
    final delegate = b.delegate, accel = b.accelerator;
    try {
      interpreter.allocateTensors();
    } catch (_) {
      b.close();
      rethrow;
    }
    final inShape = interpreter.getInputTensor(0).shape;
    final outShape = interpreter.getOutputTensor(0).shape;
    if (!_sameShape(inShape, spec.inputShape) ||
        !_sameShape(outShape, spec.outputShape)) {
      b.close();
      throw StateError(
        'Model shape $inShape → $outShape does not match spec '
        '${spec.inputShape} → ${spec.outputShape}',
      );
    }
    return _InterpreterRunner(
      interpreter,
      options,
      delegate,
      accel,
      Float32List(spec.outputElements),
    );
  }

  static bool _sameShape(List<int> a, List<int> b) =>
      a.length == b.length &&
      Iterable.generate(a.length).every((i) => a[i] == b[i]);

  @override
  Float32List run(Float32List input) {
    _interpreter
        .getInputTensor(0)
        .setTo(
          input.buffer.asUint8List(input.offsetInBytes, input.lengthInBytes),
        );
    _interpreter.invoke();
    _interpreter.getOutputTensor(0).copyTo(_out);
    return _out;
  }

  @override
  void close() {
    _interpreter.close();
    _options.delete();
    _delegate?.delete();
  }

  @override
  String get runtimeLabel => 'TFLite Interpreter ${Interpreter.version}';
  @override
  List<String> get requestedAccelerators => [_accel];
  @override
  List<String> get actualAccelerators =>
      _interpreter.hasActiveDelegate ? [_accel] : ['CPU (built-in kernels)'];
  @override
  bool get didFallback => !_interpreter.hasActiveDelegate;
  @override
  bool? get fullyAccelerated => null;
  @override
  String? get fallbackReason => _interpreter.hasActiveDelegate
      ? null
      : 'Delegate could not be applied; interpreter retried without it.';
}
