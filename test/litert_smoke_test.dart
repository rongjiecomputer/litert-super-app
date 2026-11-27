// Optional smoke test for the LiteRT CompiledModel FFI layer.
//
//   $env:LITERT_SMOKE_MODEL = "$HOME\.litert-lm\models\real-esrgan-x4v3\realesr_general_x4v3.tflite"
//   $env:LITERT_SMOKE_ARGS = "--gpu --runs=3"     # optional
//   flutter test test/litert_smoke_test.dart
//
// Prints every signature's tensors, which accelerator took the graph, and
// compile / run timings on zero inputs. Skipped unless LITERT_SMOKE_MODEL is
// set (it needs `py -I tools/fetch_litert_sdk.py` and a model on disk).
import 'dart:io';
import 'dart:math' as math;

import 'package:ai_litert/core/litert/compiled_graph.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final model = Platform.environment['LITERT_SMOKE_MODEL'];
  test(
    'compiles and runs a model',
    () {
      final flags = (Platform.environment['LITERT_SMOKE_ARGS'] ?? '')
          .split(' ')
          .where((a) => a.startsWith('--'))
          .toList();
      String? opt(String name) => flags
          .where((f) => f.startsWith('--$name='))
          .map((f) => f.substring(name.length + 3))
          .firstOrNull;
      final config = GraphConfig(
        gpu: flags.contains('--gpu'),
        fp16: flags.contains('--fp16'),
        cpuFallback: !flags.contains('--no-fallback'),
        cpuThreads: int.tryParse(opt('threads') ?? '') ?? 4,
      );
      final runs = int.tryParse(opt('runs') ?? '') ?? 3;

      // LITERT_SMOKE_MODEL may list several files separated by ';': all are
      // compiled first and then run in turn (as the TTS engines hold them).
      void log(String s) => print(s); // ignore: avoid_print
      final graphs = <CompiledGraph>[];
      final sw = Stopwatch()..start();
      for (final path in model!.split(';')) {
        sw
          ..reset()
          ..start();
        final graph = CompiledGraph.open(path, config);
        graphs.add(graph);
        log(
          '$path: compiled in ${sw.elapsedMilliseconds} ms; '
          'requested ${config.requested}',
        );
        log(
          'accelerator: ${graph.acceleratorLabel}  '
          'fullyAccelerated=${graph.isFullyAccelerated}',
        );
      }
      for (var round = 0; round < runs; round++) {
        for (final graph in graphs) {
          for (final key in graph.signatureKeys) {
            if (opt('sig') != null && opt('sig') != key) continue;
            final sig = graph.signature(key);
            if (round == 0) {
              log('signature "$key"');
              for (final n in sig.inputNames) {
                final t = sig.input(n);
                log('  in  ${t.spec}${t.isStaged ? ' [staged]' : ''}');
              }
              for (final n in sig.outputNames) {
                final t = sig.output(n);
                log('  out ${t.spec}${t.isStaged ? ' [staged]' : ''}');
              }
            }
            if (flags.contains('--random')) {
              // Deterministic non-zero float inputs, to compare backends.
              for (final n in sig.inputNames) {
                final t = sig.input(n);
                if (t.dtype != GraphDType.float32) continue;
                final v = t.asFloat32View();
                for (var i = 0; i < v.length; i++) {
                  v[i] = math.sin(i * 0.013 + n.length) * 0.5;
                }
              }
            }
            sw
              ..reset()
              ..start();
            sig.invoke();
            if (flags.contains('--random')) {
              for (final n in sig.outputNames) {
                final v = sig.output(n).asFloat32View();
                var sum = 0.0, maxAbs = 0.0, nans = 0;
                for (final x in v) {
                  if (x.isNaN) {
                    nans++;
                  } else {
                    sum += x;
                    maxAbs = math.max(maxAbs, x.abs());
                  }
                }
                log(
                  '  out $n sum=${sum.toStringAsFixed(3)} '
                  'maxAbs=${maxAbs.toStringAsFixed(4)} nan=$nans',
                );
              }
            }
            log(
              '  "$key" run ${round + 1}: ${sw.elapsedMicroseconds / 1000} ms',
            );
          }
        }
      }
      for (final g in graphs) {
        g.close();
      }
    },
    skip: model == null ? 'set LITERT_SMOKE_MODEL to run' : false,
    timeout: const Timeout(Duration(minutes: 10)),
  );
}
