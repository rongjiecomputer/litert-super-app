// End-to-end check of the real LiteRT runtime on the host device:
//
//   flutter test integration_test/litert_pipeline_test.dart -d windows \
//     --dart-define=SAMPLE_IMAGE=/path/to/image.png \
//     --dart-define=SAMPLE_VIDEO=/path/to/clip.mp4 //     --dart-define=VIDEO_FRAMES=60 --dart-define=OUT_DIR=/tmp/out
//
// Requires the Real-ESRGAN model in ~/.litert-lm/models (download it from
// the app's Models page). Video is skipped when ffmpeg is not found.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ai_litert/core/inference/inference_options.dart';
import 'package:ai_litert/core/inference/inference_worker.dart';
import 'package:ai_litert/core/metrics/run_metrics.dart';
import 'package:ai_litert/core/models/model_catalog.dart';
import 'package:ai_litert/core/paths.dart';
import 'package:ai_litert/features/upscale/ffmpeg.dart';
import 'package:ai_litert/features/upscale/upscale_tasks.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;

const sampleImage = String.fromEnvironment('SAMPLE_IMAGE');
const sampleVideo = String.fromEnvironment('SAMPLE_VIDEO');
const videoFrames = int.fromEnvironment('VIDEO_FRAMES');
const outDirDefine = String.fromEnvironment('OUT_DIR');

String get outDir =>
    outDirDefine.isNotEmpty ? outDirDefine : Directory.systemTemp.path;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const spec = ModelCatalog.realEsrganX4v3;
  late String modelPath;
  late InferenceWorker worker;
  final caps = PlatformCaps.current();
  final outputs = <String, Uint8List>{};

  setUpAll(() async {
    final paths = await AppPaths.resolve();
    modelPath = p.join(paths.modelDir(spec.id), spec.fileName);
    expect(File(modelPath).existsSync(), isTrue, reason: 'model missing');
    expect(sampleImage, isNotEmpty, reason: 'pass --dart-define=SAMPLE_IMAGE');
    worker = await InferenceWorker.spawn();
  });

  tearDownAll(() => worker.dispose());

  final configs = <String, InferenceOptions>{
    'cm-cpu': const InferenceOptions(accelerator: AcceleratorKind.cpu),
    'cm-cpu-hostbuf': const InferenceOptions(
      accelerator: AcceleratorKind.cpu,
      bufferMode: BufferModeKind.hostMemory,
    ),
    'cm-gpu-fp32': const InferenceOptions(cpuFallback: false),
    'cm-gpu-fp16': const InferenceOptions(
      precision: PrecisionKind.fp16,
      cpuFallback: false,
    ),
    'interp-xnnpack-4t': const InferenceOptions(
      runtime: RuntimeKind.interpreter,
      accelerator: AcceleratorKind.cpu,
      cpuThreads: 4,
    ),
  };

  for (final MapEntry(key: name, value: options) in configs.entries) {
    testWidgets('image upscale · $name', (tester) async {
      if (!caps.accelerators(options.runtime).contains(options.accelerator)) {
        markTestSkipped('$name not supported on ${caps.os}');
        return;
      }
      await tester.runAsync(() async {
        final load = await worker.run(LoadModelTask(modelPath, spec, options));
        final r = await worker.run(UpscaleImageTask(sampleImage, 8));
        final stats = LatencyStats.of(r.inferMs);
        // ignore: avoid_print
        print(
          '[$name] ${load.actual.join('+')} fallback=${load.didFallback} '
          'load=${formatMs(load.loadMs)} warmup=${formatMs(load.warmupMs)} '
          'tiles=${r.tiles} p50=${formatMs(stats.p50)} '
          'p90=${formatMs(stats.p90)} total=${formatMs(r.totalMs)} '
          'rss=${formatBytes(ProcessInfo.currentRss)}',
        );
        expect(r.outputSize, (r.inputSize.$1 * 4, r.inputSize.$2 * 4));
        final px = r.outputPreview.materialize();
        expect(px.length, r.outputSize.$1 * r.outputSize.$2 * 4);
        // Output must not be blank.
        final mean = px.fold<int>(0, (a, b) => a + b) / px.length;
        expect(mean, inInclusiveRange(10, 250));
        outputs[name] = px;
        if (name == 'cm-cpu') {
          final out = p.join(outDir, 'sample_x4.png');
          final ms = await worker.run(SaveLastImageTask(out));
          expect(File(out).existsSync(), isTrue);
          // ignore: avoid_print
          print('[$name] saved $out in ${formatMs(ms)}');
        }
      });
    }, timeout: const Timeout(Duration(minutes: 5)));
  }

  test('backends agree with CPU reference (PSNR)', () {
    final ref = outputs['cm-cpu'];
    if (ref == null) return;
    for (final e in outputs.entries.where((e) => e.key != 'cm-cpu')) {
      final psnr = _psnr(ref, e.value);
      // ignore: avoid_print
      print('[psnr] ${e.key} vs cm-cpu: ${psnr.toStringAsFixed(1)} dB');
      expect(psnr, greaterThan(30), reason: e.key);
    }
  });

  testWidgets('video upscale via ffmpeg', (tester) async {
    await tester.runAsync(() async {
      final tools = await FfmpegTools.locate();
      if (tools == null || sampleVideo.isEmpty) {
        markTestSkipped('ffmpeg or SAMPLE_VIDEO missing');
        return;
      }
      final info = await tools.probe(sampleVideo);
      final (w, h) = VideoJob.decodeSize(info, 1.0);
      final out = p.join(outDir, 'sample_x4.mp4');
      await worker.run(
        LoadModelTask(modelPath, spec, const InferenceOptions()),
      );
      final progress = <VideoProgress>[];
      final r = await worker.run(
        UpscaleVideoTask(
          VideoJob(
            ffmpeg: tools.ffmpeg,
            inputPath: sampleVideo,
            outputPath: out,
            decodeWidth: w,
            decodeHeight: h,
            fpsExpr: info.fpsExpr,
            maxFrames: videoFrames,
            hasAudio: info.hasAudio,
            codecArgs: await tools.videoCodecArgs(),
            overlap: 8,
          ),
        ),
        onProgress: (m) => progress.add(m as VideoProgress),
      );
      // ignore: avoid_print
      print(
        '[video] ${r.frames} frames ${r.inputSize} → ${r.outputSize} '
        'in ${formatMs(r.totalMs)} '
        '(${(r.frames / (r.totalMs / 1000)).toStringAsFixed(2)} fps, '
        'encode ${formatMs(r.encodeMs)}, io ${formatMs(r.ioMs)})',
      );
      final expected = videoFrames > 0
          ? math.min(videoFrames, info.estimatedFrames!)
          : info.estimatedFrames;
      expect(r.frames, expected);
      expect(progress, isNotEmpty);
      final outInfo = await tools.probe(out);
      expect((outInfo.width, outInfo.height), (w * 4, h * 4));
      expect(outInfo.hasAudio, info.hasAudio);
      expect(outInfo.estimatedFrames, closeTo(r.frames, 1));
    });
  }, timeout: const Timeout(Duration(minutes: 10)));
}

double _psnr(Uint8List a, Uint8List b) {
  var se = 0.0;
  var n = 0;
  for (var i = 0; i < a.length; i++) {
    if (i % 4 == 3) continue; // alpha
    final d = a[i] - b[i];
    se += d * d;
    n++;
  }
  final mse = se / n;
  return mse == 0 ? 99 : 10 * math.log(255 * 255 / mse) / math.ln10;
}
