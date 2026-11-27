import 'dart:typed_data';

import '../../../core/inference/inference_options.dart';
import '../dsp/audio.dart';
import '../dsp/mel.dart';
import 'graph.dart';

/// Qwen3-TTS speaker encoder (ECAPA-TDNN) exported by
/// `tools/tts/export_qwen3_speaker_encoder.py`: 24 kHz audio → log-mel →
/// 1024-d x-vector used as a Qwen3-TTS voice.
///
/// The graph has fixed-length signatures `frames_<T>`; the reference mel is
/// center-cropped to the largest bucket that fits (tiled when the clip is
/// shorter than the smallest), and long clips average several windows.
class SpeakerEncoder {
  SpeakerEncoder._(this._graph, this._buckets);

  final LiteRtGraph _graph;
  final List<int> _buckets; // ascending frame counts
  final _mel = MelSpectrogram();

  static const sampleRate = 24000;
  static const melBins = 128;

  /// Shortest / longest reference audio the buckets cover.
  double get minSeconds => _buckets.first * 256 / sampleRate;
  double get maxSeconds => _buckets.last * 256 / sampleRate;

  static SpeakerEncoder load(String path, InferenceOptions o) {
    final g = LiteRtGraph.open(path, o);
    final buckets = <int>[
      for (final k in g.signatureKeys)
        if (k.startsWith('frames_')) int.parse(k.substring(7)),
    ]..sort();
    if (buckets.isEmpty) {
      g.close();
      throw StateError(
        'speaker_encoder.tflite has no frames_<T> signatures; re-export it '
        'with tools/tts/export_qwen3_speaker_encoder.py',
      );
    }
    return SpeakerEncoder._(g, buckets);
  }

  /// Computes the x-vector for [audio] (any rate; resampled to 24 kHz).
  Float32List embed(PcmAudio audio) {
    var pcm = audio.sampleRate == sampleRate
        ? audio.samples
        : resample(audio.samples, audio.sampleRate, sampleRate);
    pcm = trimSilence(pcm, sampleRate: sampleRate);
    final mel = _mel.compute(pcm); // [frames][128]
    final frames = mel.length ~/ melBins;
    if (frames == 0) throw ArgumentError('Reference audio is empty');

    final t = _buckets.lastWhere((b) => b <= frames, orElse: () => _buckets.first);
    final windows = <int>[]; // start frames
    if (frames <= t) {
      windows.add(0);
    } else {
      final n = (frames / t).floor().clamp(1, 4);
      final span = frames - t;
      for (var i = 0; i < n; i++) {
        windows.add(n == 1 ? span ~/ 2 : (span * i / (n - 1)).round());
      }
    }
    final sig = _graph.sig('frames_$t');
    final inName = sig.inputNames.first;
    final acc = Float64List(1024);
    for (final start in windows) {
      final input = sig.inF(inName);
      for (var f = 0; f < t; f++) {
        // Tile short clips: frame index wraps around.
        final src = (start + f) % frames;
        input.setRange(f * melBins, (f + 1) * melBins, mel, src * melBins);
      }
      sig.invoke();
      final out = sig.outF(sig.outputNames.first);
      for (var i = 0; i < acc.length; i++) {
        acc[i] += out[i];
      }
    }
    return Float32List.fromList([for (final v in acc) v / windows.length]);
  }

  void close() => _graph.close();
}
