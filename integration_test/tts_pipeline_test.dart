// End-to-end check of the TTS engines on the real LiteRT runtime:
//
//   flutter test integration_test/tts_pipeline_test.dart -d windows \
//     --dart-define=OUT_DIR=/tmp/out [--dart-define=TTS_ENGINES=kokoro,qwen3]
//
// Requires the Kokoro-82M / Qwen3-TTS models in ~/.litert-lm/models
// (download them from the app's Models page); missing models are skipped.
// Writes one WAV per engine to OUT_DIR for listening. The speaker-encoder
// parity check runs when speaker_encoder.tflite and its fixture exist.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ai_litert/core/inference/inference_options.dart';
import 'package:ai_litert/core/inference/inference_worker.dart';
import 'package:ai_litert/core/models/model_catalog.dart';
import 'package:ai_litert/core/paths.dart';
import 'package:ai_litert/features/tts/dsp/audio.dart';
import 'package:ai_litert/features/tts/dsp/mel.dart';
import 'package:ai_litert/features/tts/engines/tts_engine.dart';
import 'package:ai_litert/features/tts/text/npy.dart';
import 'package:ai_litert/features/tts/tts_tasks.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;

const outDirDefine = String.fromEnvironment('OUT_DIR');
const enginesDefine = String.fromEnvironment('TTS_ENGINES', defaultValue: 'kokoro,qwen3');

String get outDir => outDirDefine.isNotEmpty ? outDirDefine : Directory.systemTemp.path;

double rms(Float32List x) {
  var s = 0.0;
  for (final v in x) {
    s += v * v;
  }
  return x.isEmpty ? 0 : math.sqrt(s / x.length);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  late AppPaths paths;
  late InferenceWorker worker;
  final caps = PlatformCaps.current();
  // TTS_ACCEL=gpu runs the graphs on the GPU (default: CPU reference).
  const accel = String.fromEnvironment('TTS_ACCEL', defaultValue: 'cpu');
  final options = InferenceOptions(
    accelerator: AcceleratorKind.values.firstWhere((a) => a.name == accel),
    cpuThreads: math.min(8, caps.maxThreads),
  ).sanitized(caps);

  setUpAll(() async {
    paths = await AppPaths.resolve();
    worker = await InferenceWorker.spawn();
  });
  tearDownAll(() => worker.dispose());

  Future<Float32List> synth(TtsEngineKind kind, String dir, TtsRequest req, String name) async {
    final load = await worker.run(LoadTtsEngineTask(kind, dir, options));
    // ignore: avoid_print
    print('[$name] load ${load.info.loadMs.round()} ms, warm-up '
        '${load.info.warmupMs.round()} ms, ${load.voices.length} voices');
    final chunks = <Float32List>[];
    final stats = await worker.run(
      SynthesizeTask(req),
      onProgress: (m) {
        if (m is TtsAudioChunk) chunks.add(m.materialize());
      },
    );
    final all = Float32List(chunks.fold(0, (a, c) => a + c.length));
    var o = 0;
    for (final c in chunks) {
      all.setAll(o, c);
      o += c.length;
    }
    // ignore: avoid_print
    print('[$name] ${stats.audioSeconds.toStringAsFixed(2)} s audio in '
        '${(stats.totalMs / 1000).toStringAsFixed(1)} s (RTF ${stats.rtf.toStringAsFixed(2)}), '
        'first audio ${(stats.firstAudioMs / 1000).toStringAsFixed(1)} s, '
        '${stats.units} ${stats.unitLabel}s, stages ${stats.stagesMs.map((k, v) => MapEntry(k, v.round()))}');
    Directory(outDir).createSync(recursive: true);
    File(p.join(outDir, '$name.wav')).writeAsBytesSync(encodeWav16(all, 24000));
    expect(all.length, stats.samples);
    return all;
  }

  String kokoroDir() => paths.modelDir(ModelCatalog.kokoro82m.id);
  String qwenDir() => paths.modelDir(ModelCatalog.qwen3Tts06b.id);

  test('Kokoro synthesizes English text', () async {
    if (!enginesDefine.contains('kokoro') ||
        !File(p.join(kokoroDir(), 'vocoder.tflite')).existsSync()) {
      markTestSkipped('Kokoro not downloaded');
      return;
    }
    const text = 'Hello world! This is a test of Kokoro, running on LiteRT. '
        'It costs \$5.99 and was released in 2025.';
    final pcm = await synth(
      TtsEngineKind.kokoro,
      kokoroDir(),
      const TtsRequest(text: text, voiceId: 'af_heart', seed: 1),
      'kokoro_af_heart',
    );
    expect(pcm.length / 24000, inInclusiveRange(4.0, 15.0));
    expect(rms(pcm), greaterThan(0.01));
  }, timeout: const Timeout(Duration(minutes: 10)));

  test('Qwen3-TTS synthesizes (greedy, demo voice)', () async {
    if (!enginesDefine.contains('qwen3') ||
        !File(p.join(qwenDir(), 'talker_int4.tflite')).existsSync()) {
      markTestSkipped('Qwen3-TTS not downloaded');
      return;
    }
    final pcm = await synth(
      TtsEngineKind.qwen3,
      qwenDir(),
      const TtsRequest(
        text: 'Hello, this is Qwen three speaking on device.',
        voiceId: 'demo',
        language: 'english',
        greedy: true,
      ),
      'qwen3_demo',
    );
    expect(pcm.length / 24000, inInclusiveRange(1.5, 12.0));
    expect(rms(pcm), greaterThan(0.01));
  }, timeout: const Timeout(Duration(minutes: 20)));

  test('Speaker encoder matches the PyTorch fixture', () async {
    final enc = p.join(qwenDir(), 'speaker_encoder.tflite');
    final fixture = File(p.join('test', 'goldens', 'tts', 'speaker_encoder_fixture.json'));
    if (!File(enc).existsSync() || !fixture.existsSync()) {
      markTestSkipped('speaker_encoder.tflite or fixture missing');
      return;
    }
    final fx = jsonDecode(fixture.readAsStringSync()) as Map<String, dynamic>;
    final expected = (fx['xvector'] as List).cast<num>();
    final seconds = (fx['seconds'] as num).toDouble();
    final frames = fx['frames'] as int;
    final y = speakerTestSignal(seconds);
    // Front end parity on sampled mel values.
    final mel = MelSpectrogram().compute(y);
    var maxErr = 0.0;
    for (final s in (fx['mel_samples'] as List).cast<List>()) {
      final v = mel[(s[0] as int) * 128 + (s[1] as int)];
      maxErr = math.max(maxErr, (v - (s[2] as num)).abs());
    }
    expect(maxErr, lessThan(5e-3), reason: 'mel mismatch');
    // Encode exactly the fixture's frames (crop the audio accordingly).
    final samples = (frames - 1) * 256 + 256;
    final xv = await worker.run(
      CloneVoiceTask(enc, options, Float32List.sublistView(y, 0, samples), 24000),
    );
    var dot = 0.0, na = 0.0, nb = 0.0;
    for (var i = 0; i < expected.length; i++) {
      dot += xv[i] * expected[i];
      na += xv[i] * xv[i];
      nb += expected[i] * expected[i];
    }
    final cos = dot / math.sqrt(na * nb);
    // ignore: avoid_print
    print('[speaker] cosine vs torch = ${cos.toStringAsFixed(5)}');
    expect(cos, greaterThan(0.99));
  });

  test('Cloned voices: demo round-trips, a new voice synthesizes', () async {
    final enc = p.join(qwenDir(), 'speaker_encoder.tflite');
    final demoWav = File(p.join(outDir, 'qwen3_demo.wav'));
    final kokoroWav = File(p.join(outDir, 'kokoro_af_heart.wav'));
    if (!File(enc).existsSync() || !demoWav.existsSync() || !kokoroWav.existsSync()) {
      markTestSkipped('needs speaker_encoder.tflite and the WAVs from the engine tests');
      return;
    }
    Future<Float32List> clone(File wav) {
      final a = decodeWav(wav.readAsBytesSync());
      return worker.run(CloneVoiceTask(enc, options, a.samples, a.sampleRate));
    }

    final demo = NpyArray.readFile(p.join(qwenDir(), 'voices', 'demo_speaker.npy')).data;
    final fromDemo = await clone(demoWav);
    final fromKokoro = await clone(kokoroWav);
    final cDemo = cosine(fromDemo, demo), cOther = cosine(fromKokoro, demo);
    final dDemo = relDistance(fromDemo, demo), dOther = relDistance(fromKokoro, demo);
    // ignore: avoid_print
    print('[clone] demo audio vs demo x-vector: cos ${cDemo.toStringAsFixed(3)}, '
        'rel. dist ${dDemo.toStringAsFixed(3)}; kokoro audio vs demo: cos '
        '${cOther.toStringAsFixed(3)}, rel. dist ${dOther.toStringAsFixed(3)}');
    // Speech generated with the demo voice maps back closer to its x-vector
    // than a different speaker does (raw x-vector cosines are all high).
    expect(cDemo, greaterThan(cOther));
    expect(dDemo, lessThan(dOther));

    final pcm = await synth(
      TtsEngineKind.qwen3,
      qwenDir(),
      TtsRequest(
        text: 'This voice was cloned on device from a short clip.',
        voiceId: 'custom',
        language: 'english',
        greedy: true,
        speakerEmbedding: fromKokoro,
      ),
      'qwen3_cloned_from_kokoro',
    );
    expect(rms(pcm), greaterThan(0.01));
  }, timeout: const Timeout(Duration(minutes: 20)));
}

/// |a - b| / |b|
double relDistance(Float32List a, Float32List b) {
  var d = 0.0, nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    d += (a[i] - b[i]) * (a[i] - b[i]);
    nb += b[i] * b[i];
  }
  return math.sqrt(d / nb);
}

double cosine(Float32List a, Float32List b) {
  var dot = 0.0, na = 0.0, nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / math.sqrt(na * nb);
}

/// Same deterministic signal as `test_signal()` in
/// tools/tts/export_qwen3_speaker_encoder.py, minus the tiny noise term.
Float32List speakerTestSignal(double seconds) {
  const sr = 24000;
  final n = (seconds * sr).round();
  final y = Float32List(n);
  var phase = 0.0;
  for (var i = 0; i < n; i++) {
    final t = i / sr;
    final f0 = 140 + 25 * math.sin(2 * math.pi * 3 * t);
    phase += 2 * math.pi * f0 / sr;
    var v = 0.0;
    for (var h = 1; h <= 8; h++) {
      v += 0.3 / h * math.sin(h * phase);
    }
    y[i] = v * (0.6 + 0.4 * math.sin(2 * math.pi * 0.5 * t));
  }
  return y;
}
