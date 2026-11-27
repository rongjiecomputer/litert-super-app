import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import '../metrics/run_metrics.dart';
import '../models/model_spec.dart';
import 'inference_options.dart';
import 'model_runner.dart';

/// Work executed inside the inference isolate, next to the loaded model.
///
/// Subclasses must only hold sendable fields (primitives, lists, typed data,
/// plain objects); they are copied to the worker isolate. Use cases add
/// their own tasks; the worker itself stays generic.
abstract class WorkerTask<R> {
  const WorkerTask();
  Future<R> run(WorkerContext ctx);
}

/// A multi-graph pipeline (e.g. a TTS engine) owned by the worker isolate.
///
/// Use cases whose models don't fit [ModelRunner]'s single float32
/// input/output shape load one of these instead; only one engine (or one
/// runner) is resident at a time.
abstract class WorkerEngine {
  void close();
}

/// State visible to tasks inside the worker isolate.
class WorkerContext {
  WorkerContext._(this._send);

  final void Function(Object progress) _send;
  ModelRunner? runner;
  ModelSpec? spec;
  WorkerEngine? engine;
  bool _cancelRequested = false;

  /// Scratch space that survives between tasks (e.g. last result to save).
  final Map<String, Object?> scratch = {};

  bool get isCancelled => _cancelRequested;

  void progress(Object payload) => _send(payload);

  ModelRunner requireRunner() =>
      runner ?? (throw StateError('No model loaded in worker'));

  /// Returns the loaded engine of type [T] or throws.
  T requireEngine<T extends WorkerEngine>() {
    final e = engine;
    if (e is T) return e;
    throw StateError('No ${T.toString()} loaded in worker');
  }

  /// Closes whatever model or engine is resident.
  void unloadAll() {
    runner?.close();
    runner = null;
    spec = null;
    engine?.close();
    engine = null;
  }

  /// Lets queued messages (notably cancel requests) through, then throws
  /// [WorkerCancelled] if cancellation was requested. Long synchronous
  /// loops should call this between steps.
  Future<void> checkpoint() async {
    await Future<void>.delayed(Duration.zero);
    if (_cancelRequested) throw const WorkerCancelled();
  }
}

/// Loads (or reloads) the model and runs one warm-up inference.
class LoadModelTask extends WorkerTask<LoadInfo> {
  const LoadModelTask(this.modelPath, this.spec, this.options);

  final String modelPath;
  final ModelSpec spec;
  final InferenceOptions options;

  @override
  Future<LoadInfo> run(WorkerContext ctx) async {
    ctx.unloadAll();
    final rssBefore = ProcessInfo.currentRss;
    final sw = Stopwatch()..start();
    final runner = ModelRunner.load(modelPath, spec, options);
    final loadMs = sw.elapsedMicroseconds / 1000;
    sw
      ..reset()
      ..start();
    // First inference pays for lazy allocation / shader compilation.
    runner.run(Float32List(spec.inputElements));
    final warmupMs = sw.elapsedMicroseconds / 1000;
    ctx.runner = runner;
    ctx.spec = spec;
    return LoadInfo(
      optionsSummary: options.summary,
      runtimeLabel: runner.runtimeLabel,
      requested: runner.requestedAccelerators,
      actual: runner.actualAccelerators,
      didFallback: runner.didFallback,
      fullyAccelerated: runner.fullyAccelerated,
      loadMs: loadMs,
      warmupMs: warmupMs,
      rssDeltaBytes: ProcessInfo.currentRss - rssBefore,
      fallbackReason: runner.fallbackReason,
    );
  }
}

class UnloadModelTask extends WorkerTask<void> {
  const UnloadModelTask();
  @override
  Future<void> run(WorkerContext ctx) async => ctx.unloadAll();
}

class WorkerCancelled implements Exception {
  const WorkerCancelled();
  @override
  String toString() => 'Cancelled';
}

/// Main-isolate handle to a long-lived background isolate that owns one
/// LiteRT model. Keeps the UI responsive while inference blocks.
class InferenceWorker {
  InferenceWorker._(this._isolate, this._toWorker, this._fromWorker);

  final Isolate _isolate;
  final SendPort _toWorker;
  final ReceivePort _fromWorker;
  final _pending = <int, _Pending>{};
  var _nextId = 0;
  var _disposed = false;

  static Future<InferenceWorker> spawn() async {
    final rp = ReceivePort();
    final ready = Completer<SendPort>();
    InferenceWorker? self;
    rp.listen((m) {
      if (m is SendPort && !ready.isCompleted) {
        ready.complete(m);
      } else {
        self?._onMessage(m);
      }
    });
    final isolate = await Isolate.spawn(
      _workerMain,
      rp.sendPort,
      debugName: 'litert-worker',
    );
    return self = InferenceWorker._(isolate, await ready.future, rp);
  }

  Future<R> run<R>(
    WorkerTask<R> task, {
    void Function(Object progress)? onProgress,
  }) {
    if (_disposed) return Future.error(StateError('Worker disposed'));
    final id = _nextId++;
    final p = _Pending(onProgress);
    _pending[id] = p;
    _toWorker.send(_TaskMsg(id, task));
    return p.completer.future.then((v) => v as R);
  }

  /// Requests cooperative cancellation of the running task.
  void cancel() {
    if (!_disposed) _toWorker.send(const _CancelMsg());
  }

  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _toWorker.send(const _DisposeMsg());
    await Future<void>.delayed(const Duration(milliseconds: 200));
    _isolate.kill(priority: Isolate.beforeNextEvent);
    _fromWorker.close();
    for (final p in _pending.values) {
      p.completer.completeError(StateError('Worker disposed'));
    }
    _pending.clear();
  }

  void _onMessage(Object? m) {
    switch (m) {
      case _ProgressMsg(:final id, :final payload):
        _pending[id]?.onProgress?.call(payload);
      case _DoneMsg(:final id, :final result):
        _pending.remove(id)?.completer.complete(result);
      case _FailedMsg(:final id, :final error, :final stack, :final cancelled):
        _pending
            .remove(id)
            ?.completer
            .completeError(
              cancelled ? const WorkerCancelled() : _RemoteError(error),
              StackTrace.fromString(stack),
            );
    }
  }
}

class _Pending {
  _Pending(this.onProgress);
  final completer = Completer<Object?>();
  final void Function(Object progress)? onProgress;
}

class _RemoteError implements Exception {
  _RemoteError(this.message);
  final String message;
  @override
  String toString() => message;
}

// ---- Wire protocol ---------------------------------------------------------

class _TaskMsg {
  const _TaskMsg(this.id, this.task);
  final int id;
  final WorkerTask<Object?> task;
}

class _CancelMsg {
  const _CancelMsg();
}

class _DisposeMsg {
  const _DisposeMsg();
}

class _ProgressMsg {
  const _ProgressMsg(this.id, this.payload);
  final int id;
  final Object payload;
}

class _DoneMsg {
  const _DoneMsg(this.id, this.result);
  final int id;
  final Object? result;
}

class _FailedMsg {
  const _FailedMsg(this.id, this.error, this.stack, this.cancelled);
  final int id;
  final String error;
  final String stack;
  final bool cancelled;
}

// ---- Worker isolate --------------------------------------------------------

void _workerMain(SendPort out) {
  final inbox = ReceivePort();
  out.send(inbox.sendPort);
  var currentId = -1;
  final ctx = WorkerContext._((p) => out.send(_ProgressMsg(currentId, p)));
  var queue = Future<void>.value();

  inbox.listen((m) {
    switch (m) {
      case _CancelMsg():
        ctx._cancelRequested = true;
      case _DisposeMsg():
        ctx.unloadAll();
        inbox.close();
      case _TaskMsg(:final id, :final task):
        queue = queue.then((_) async {
          currentId = id;
          ctx._cancelRequested = false;
          try {
            final result = await task.run(ctx);
            out.send(_DoneMsg(id, result));
          } catch (e, st) {
            out.send(_FailedMsg(id, '$e', '$st', e is WorkerCancelled));
          }
        });
    }
  });
}
