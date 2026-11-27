import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import '../../core/inference/inference_options.dart';
import '../../core/inference/inference_worker.dart';
import '../../core/metrics/run_metrics.dart';
import '../../core/models/model_catalog.dart';
import '../../core/models/model_manager.dart';
import '../../core/models/model_spec.dart';
import '../../core/settings.dart';
import '../upscale/ffmpeg.dart';
import 'audio_player.dart';
import 'dsp/audio.dart';
import 'engines/kokoro_engine.dart';
import 'engines/qwen3_tts_engine.dart';
import 'engines/tts_engine.dart';
import 'text/npy.dart';
import 'tts_tasks.dart';
import 'voice_recorder.dart';

enum TtsBusy { idle, loading, synthesizing, cloning }

/// A user-cloned Qwen3-TTS voice (x-vector saved as `<name>.npy`).
class CustomVoice {
  const CustomVoice(this.name, this.path, this.embedding);
  final String name;
  final String path;
  final Float32List embedding;

  String get id => 'custom:$name';
  TtsVoice get voice => TtsVoice(id, name, group: 'Cloned', custom: true);
}

/// One finished (or cancelled) synthesis, for the metrics panel.
class TtsRun {
  const TtsRun({
    required this.at,
    required this.engine,
    required this.voice,
    required this.text,
    required this.optionsSummary,
    required this.accelerator,
    required this.stats,
    required this.rssPeak,
    required this.rssDelta,
    this.loadMs,
    this.cancelled = false,
  });

  final DateTime at;
  final TtsEngineKind engine;
  final String voice;
  final String text;
  final String optionsSummary;
  final String accelerator;
  final TtsStats stats;
  final int rssPeak, rssDelta;
  final double? loadMs;
  final bool cancelled;
}

class TtsController extends ChangeNotifier {
  TtsController({
    required this.settings,
    required this.models,
    required this.caps,
    required this.voicesDir,
    required String tempDir,
  }) : recorder = VoiceRecorder(tempDir) {
    options = settings.inferenceOptions(useCaseId, caps);
    final pr = settings.useCaseParams(useCaseId);
    engine = TtsEngineKind.values.firstWhere(
      (e) => e.name == pr['engine'],
      orElse: () => TtsEngineKind.kokoro,
    );
    kokoroVoice = pr['kokoroVoice'] as String? ?? 'af_heart';
    qwenVoice = pr['qwenVoice'] as String? ?? Qwen3TtsEngine.demoVoice.id;
    language = pr['language'] as String? ?? 'auto';
    speed = (pr['speed'] as num?)?.toDouble() ?? 1.0;
    temperature = (pr['temperature'] as num?)?.toDouble() ?? 0.9;
    topK = (pr['topK'] as num?)?.toInt() ?? 50;
    greedy = pr['greedy'] as bool? ?? false;
    text = pr['text'] as String? ?? sampleText[engine]!;
    _loadCustomVoices();
  }

  static const useCaseId = 'tts';

  static const sampleText = {
    TtsEngineKind.kokoro:
        'Hello! This is Kokoro, an 82 million parameter text-to-speech model, '
        'running entirely on your device with LiteRT.',
    TtsEngineKind.qwen3:
        'Hello! This is Qwen3 TTS speaking on device. '
        '它也会说中文，还有日语、韩语和其他七种语言。',
  };

  final AppSettings settings;
  final ModelManager models;
  final PlatformCaps caps;
  final String voicesDir;
  final player = TtsPlayer();
  final VoiceRecorder recorder;

  // ---- configuration ---------------------------------------------------------
  late InferenceOptions options;
  late TtsEngineKind engine;
  late String kokoroVoice, qwenVoice, language, text;
  late double speed, temperature;
  late int topK;
  late bool greedy;

  ModelSpec specFor(TtsEngineKind k) => switch (k) {
    TtsEngineKind.kokoro => ModelCatalog.kokoro82m,
    TtsEngineKind.qwen3 => ModelCatalog.qwen3Tts06b,
  };
  ModelSpec get spec => specFor(engine);
  bool get modelReady => models.isReady(spec);

  String get voiceId => engine == TtsEngineKind.kokoro ? kokoroVoice : qwenVoice;

  // ---- loaded engine -----------------------------------------------------------
  TtsEngineKind? loadedEngine;
  InferenceOptions? loadedOptions;
  LoadInfo? loadInfo;
  List<TtsVoice>? _engineVoices;
  bool get needsReload => loadedEngine != engine || loadedOptions != options;
  bool get loaded => loadInfo != null && loadedEngine == engine;

  final List<CustomVoice> customVoices = [];

  List<TtsVoice> get voices {
    final base = loadedEngine == engine && _engineVoices != null
        ? _engineVoices!
        : switch (engine) {
            TtsEngineKind.kokoro => [
              for (final v in KokoroVoices.english) KokoroEngine.voiceInfo(v),
            ],
            TtsEngineKind.qwen3 => const [Qwen3TtsEngine.demoVoice],
          };
    return [
      ...base,
      if (engine == TtsEngineKind.qwen3) for (final c in customVoices) c.voice,
    ];
  }

  // ---- job state -----------------------------------------------------------
  TtsBusy busy = TtsBusy.idle;
  String? status;
  String? error;
  double generatedSeconds = 0;
  bool get isBusy => busy != TtsBusy.idle;

  // ---- voice cloning --------------------------------------------------------
  PcmAudio? reference;
  String? referenceName;
  bool get hasSpeakerEncoder =>
      models.hasFile(ModelCatalog.qwen3Tts06b, Qwen3TtsEngine.speakerEncoderFile);

  // ---- metrics ---------------------------------------------------------------
  TtsRun? lastRun;
  final List<TtsRun> history = [];

  InferenceWorker? _worker;
  final _rss = RssSampler();
  bool _disposed = false;

  void _saveParams() => settings.setUseCaseParams(useCaseId, {
    'engine': engine.name,
    'kokoroVoice': kokoroVoice,
    'qwenVoice': qwenVoice,
    'language': language,
    'speed': speed,
    'temperature': temperature,
    'topK': topK,
    'greedy': greedy,
    'text': text,
  });

  void setOptions(InferenceOptions o) {
    options = o.sanitized(caps);
    settings.setInferenceOptions(useCaseId, options);
    _notify();
  }

  void setEngine(TtsEngineKind k) {
    if (k == engine) return;
    // Swap in the other engine's sample if the text is still a sample.
    if (sampleText.values.contains(text)) text = sampleText[k]!;
    engine = k;
    _saveParams();
    _notify();
  }

  void setParams({
    String? voice,
    String? language,
    double? speed,
    double? temperature,
    int? topK,
    bool? greedy,
    String? text,
  }) {
    if (voice != null) {
      if (engine == TtsEngineKind.kokoro) {
        kokoroVoice = voice;
      } else {
        qwenVoice = voice;
      }
    }
    this.language = language ?? this.language;
    this.speed = speed ?? this.speed;
    this.temperature = temperature ?? this.temperature;
    this.topK = topK ?? this.topK;
    this.greedy = greedy ?? this.greedy;
    this.text = text ?? this.text;
    _saveParams();
    _notify();
  }

  // ---- engine lifecycle ------------------------------------------------------

  Future<InferenceWorker> _ensureWorker() async =>
      _worker ??= await InferenceWorker.spawn();

  Future<double?> _ensureLoaded() async {
    if (!modelReady) throw StateError('Download ${spec.name} first');
    if (!needsReload && loadInfo != null) return null;
    final w = await _ensureWorker();
    final kind = engine, requested = options;
    status = 'Loading ${kind.label} (${requested.summary}) and warming up…';
    _notify();
    try {
      final r = await w.run(
        LoadTtsEngineTask(kind, models.paths.modelDir(specFor(kind).id), requested),
      );
      loadInfo = r.info;
      _engineVoices = r.voices;
      loadedEngine = kind;
      loadedOptions = requested;
    } catch (_) {
      loadInfo = null;
      loadedEngine = null;
      loadedOptions = null;
      rethrow;
    }
    return loadInfo!.loadMs + loadInfo!.warmupMs;
  }

  Future<void> loadModel() => _guard(TtsBusy.loading, () async {
    loadedOptions = null; // force reload
    _rss.start();
    await _ensureLoaded();
    _rss.stop();
    status = '${engine.label} loaded on ${loadInfo!.actual.join(' + ')}';
  });

  Future<void> unloadModel() async {
    if (isBusy || _worker == null) return;
    await _worker!.run(const UnloadModelTask());
    loadInfo = null;
    loadedEngine = null;
    loadedOptions = null;
    status = 'Model unloaded';
    _notify();
  }

  void cancel() {
    _worker?.cancel();
    status = 'Stopping…';
    _notify();
  }

  // ---- synthesis -------------------------------------------------------------

  Future<void> synthesize() async {
    final input = text.trim();
    if (input.isEmpty) return;
    await _guard(TtsBusy.synthesizing, () async {
      _rss.start();
      final loadMs = await _ensureLoaded();
      final kind = engine;
      final custom = kind == TtsEngineKind.qwen3
          ? customVoices.where((c) => c.id == qwenVoice).firstOrNull
          : null;
      final req = TtsRequest(
        text: input,
        voiceId: kind == TtsEngineKind.kokoro ? kokoroVoice : Qwen3TtsEngine.demoVoice.id,
        language: language,
        speed: speed,
        temperature: temperature,
        topK: topK,
        greedy: greedy,
        speakerEmbedding: custom?.embedding,
      );
      await player.begin();
      generatedSeconds = 0;
      status = 'Synthesizing with ${kind.label}…';
      _notify();
      var cancelled = false;
      TtsStats? stats;
      final sw = Stopwatch()..start();
      try {
        stats = await _worker!.run(
          SynthesizeTask(req),
          onProgress: (m) {
            if (m is TtsAudioChunk) {
              player.add(m.materialize());
              generatedSeconds = player.seconds;
              status = 'Generated ${generatedSeconds.toStringAsFixed(1)} s of audio '
                  'in ${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)} s…';
              _notify();
            }
          },
        );
      } on WorkerCancelled {
        cancelled = true;
        rethrow;
      } finally {
        player.end();
        final (before, peak, _) = _rss.stop();
        final s = stats ??
            TtsStats(
              samples: (player.seconds * 24000).round(),
              sampleRate: 24000,
              totalMs: sw.elapsedMicroseconds / 1000,
              firstAudioMs: 0,
              units: 0,
              unitLabel: '',
            );
        if (stats != null || cancelled) {
          _record(
            TtsRun(
              at: DateTime.now(),
              engine: kind,
              voice: custom?.name ??
                  voices.where((v) => v.id == req.voiceId).firstOrNull?.label ??
                  req.voiceId,
              text: input,
              optionsSummary: loadedOptions?.summary ?? options.summary,
              accelerator: loadInfo?.actual.join(' + ') ?? '',
              stats: s,
              rssPeak: peak,
              rssDelta: peak - before,
              loadMs: loadMs,
              cancelled: cancelled,
            ),
          );
        }
      }
      status = 'Done: ${stats!.audioSeconds.toStringAsFixed(1)} s of audio in '
          '${formatMs(stats.totalMs)} (RTF ${stats.rtf.toStringAsFixed(2)})';
    });
  }

  Future<void> saveWav(String path) async {
    final clip = player.clip;
    if (clip.isEmpty) return;
    await File(path).writeAsBytes(encodeWav16(clip, player.sampleRate));
    status = 'Saved ${p.basename(path)}';
    _notify();
  }

  // ---- voice cloning --------------------------------------------------------

  /// Loads reference audio from a file: WAV is decoded in Dart, anything
  /// else through ffmpeg (if available).
  Future<void> setReferenceFile(String path) async {
    error = null;
    try {
      PcmAudio audio;
      if (p.extension(path).toLowerCase() == '.wav') {
        try {
          audio = decodeWav(await File(path).readAsBytes());
        } on FormatException {
          audio = await _ffmpegDecode(path);
        }
      } else {
        audio = await _ffmpegDecode(path);
      }
      if (audio.seconds < 1) throw const FormatException('Reference is shorter than 1 s');
      reference = audio;
      referenceName = p.basename(path);
    } catch (e) {
      error = 'Could not read ${p.basename(path)}: $e';
    }
    _notify();
  }

  Future<PcmAudio> _ffmpegDecode(String path) async {
    final ff = await FfmpegTools.locate(override: settings.ffmpegPath);
    if (ff == null) {
      throw StateError('Only WAV can be read without ffmpeg (set it in Settings)');
    }
    final (pcm, rate) = await ff.decodeAudio(path);
    return PcmAudio(pcm, rate);
  }

  Future<void> toggleRecording() async {
    if (recorder.recording) {
      final path = await recorder.stop();
      if (path != null) await setReferenceFile(path);
      if (reference != null) referenceName = 'Microphone recording';
    } else {
      await recorder.start();
    }
    _notify();
  }

  Future<void> previewReference() async {
    final r = reference;
    if (r == null) return;
    await player.begin();
    player.add(r.sampleRate == 24000 ? r.samples : resample(r.samples, r.sampleRate, 24000));
    player.end();
  }

  void clearReference() {
    reference = null;
    referenceName = null;
    _notify();
  }

  Future<void> importSpeakerEncoder(String path) async {
    await models.importFile(
      ModelCatalog.qwen3Tts06b,
      Qwen3TtsEngine.speakerEncoderFile,
      path,
    );
    status = 'Speaker encoder imported';
    _notify();
  }

  Future<void> cloneVoice(String name) async {
    final r = reference;
    final clean = name.trim().replaceAll(RegExp(r'[^\w\- ]'), '').trim();
    if (r == null || clean.isEmpty) return;
    await _guard(TtsBusy.cloning, () async {
      if (!hasSpeakerEncoder) {
        throw StateError('speaker_encoder.tflite is missing (see the export tool)');
      }
      final w = await _ensureWorker();
      status = 'Extracting speaker embedding…';
      _notify();
      final sw = Stopwatch()..start();
      final xv = await w.run(
        CloneVoiceTask(
          models.fileOf(ModelCatalog.qwen3Tts06b, Qwen3TtsEngine.speakerEncoderFile),
          options,
          r.samples,
          r.sampleRate,
        ),
      );
      Directory(voicesDir).createSync(recursive: true);
      final path = p.join(voicesDir, '$clean.npy');
      await File(path).writeAsBytes(encodeNpyF32(xv, [xv.length]));
      customVoices
        ..removeWhere((c) => c.name == clean)
        ..add(CustomVoice(clean, path, xv))
        ..sort((a, b) => a.name.compareTo(b.name));
      qwenVoice = 'custom:$clean';
      if (engine != TtsEngineKind.qwen3) setEngine(TtsEngineKind.qwen3);
      _saveParams();
      status = 'Cloned "$clean" in ${formatMs(sw.elapsedMicroseconds / 1000)}';
    });
  }

  Future<void> deleteVoice(CustomVoice v) async {
    final f = File(v.path);
    if (f.existsSync()) await f.delete();
    customVoices.remove(v);
    if (qwenVoice == v.id) qwenVoice = Qwen3TtsEngine.demoVoice.id;
    _saveParams();
    _notify();
  }

  void _loadCustomVoices() {
    final dir = Directory(voicesDir);
    if (!dir.existsSync()) return;
    for (final f in dir.listSync().whereType<File>()) {
      if (!f.path.endsWith('.npy')) continue;
      try {
        final a = NpyArray.readFile(f.path);
        if (a.data.length != Qwen3TtsEngine.hidden) continue;
        customVoices.add(CustomVoice(p.basenameWithoutExtension(f.path), f.path, a.data));
      } catch (_) {
        // Skip unreadable files.
      }
    }
    customVoices.sort((a, b) => a.name.compareTo(b.name));
  }

  // ---- helpers ---------------------------------------------------------------

  void _record(TtsRun r) {
    lastRun = r;
    history.insert(0, r);
    if (history.length > 50) history.removeLast();
  }

  Future<void> _guard(TtsBusy kind, Future<void> Function() body) async {
    if (isBusy) return;
    busy = kind;
    error = null;
    _notify();
    try {
      await body();
    } on WorkerCancelled {
      status = 'Stopped';
    } catch (e) {
      error = '$e';
      status = null;
    } finally {
      busy = TtsBusy.idle;
      _rss.stop();
      _notify();
    }
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    player.dispose();
    recorder.dispose();
    _worker?.dispose();
    super.dispose();
  }
}
