import 'dart:typed_data';

import '../../../core/inference/inference_worker.dart';

enum TtsEngineKind {
  kokoro('Kokoro-82M', 'Fast · English · 28 preset voices'),
  qwen3('Qwen3-TTS 0.6B', '10 languages · voice cloning');

  const TtsEngineKind(this.label, this.blurb);
  final String label;
  final String blurb;
}

/// A selectable voice. Sendable across isolates.
class TtsVoice {
  const TtsVoice(this.id, this.label, {this.group = '', this.custom = false});
  final String id;
  final String label;

  /// Grouping label in pickers, e.g. "American English · Female".
  final String group;

  /// User-cloned voice (Qwen3 x-vector) rather than a bundled preset.
  final bool custom;
}

/// One synthesis job. Sendable across isolates.
class TtsRequest {
  const TtsRequest({
    required this.text,
    required this.voiceId,
    this.language = 'auto',
    this.speed = 1.0,
    this.temperature = 0.9,
    this.topK = 50,
    this.greedy = false,
    this.seed,
    this.speakerEmbedding,
  });

  final String text;
  final String voiceId;

  /// Qwen3: `auto` or a language name (e.g. `english`).
  final String language;

  /// Kokoro speaking rate (1 = normal).
  final double speed;

  /// Qwen3 sampling.
  final double temperature;
  final int topK;
  final bool greedy;
  final int? seed;

  /// Qwen3 x-vector for a custom (cloned) voice; overrides [voiceId].
  final Float32List? speakerEmbedding;
}

/// Synthesis timing summary returned by an engine.
class TtsStats {
  const TtsStats({
    required this.samples,
    required this.sampleRate,
    required this.totalMs,
    required this.firstAudioMs,
    required this.units,
    required this.unitLabel,
    this.stagesMs = const {},
  });

  final int samples;
  final int sampleRate;
  final double totalMs;

  /// Time until the first audio chunk was emitted.
  final double firstAudioMs;

  /// Work units (Kokoro chunks / Qwen codec frames).
  final int units;
  final String unitLabel;
  final Map<String, double> stagesMs;

  double get audioSeconds => samples / sampleRate;
  double get rtf => audioSeconds == 0 ? 0 : totalMs / 1000 / audioSeconds;
}

/// A TTS pipeline resident in the inference worker.
abstract class TtsEngine implements WorkerEngine {
  TtsEngineKind get kind;
  int get sampleRate => 24000;
  List<TtsVoice> get voices;

  /// Human accelerator label for the load summary.
  String get accelerator;

  /// Runs every graph once so the first real request doesn't pay for
  /// lazy allocation (XNNPACK packing can take seconds per graph).
  Future<void> warmUp(WorkerContext ctx);

  /// Synthesizes [req], calling [emit] with each audio chunk (mono float
  /// PCM at [sampleRate]) as soon as it is ready.
  Future<TtsStats> synthesize(
    TtsRequest req,
    WorkerContext ctx,
    void Function(Float32List pcm) emit,
  );
}
