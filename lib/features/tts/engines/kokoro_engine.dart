import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import '../../../core/inference/inference_options.dart';
import '../../../core/inference/inference_worker.dart';
import '../../../core/models/model_catalog.dart' show KokoroVoices;
import '../dsp/kokoro_dsp.dart';
import '../dsp/time_stretch.dart';
import '../text/english_g2p.dart';
import 'graph.dart';
import 'tts_engine.dart';

/// Kokoro-82M (StyleTTS2 + iSTFTNet) on the two fused LiteRT-LM graphs.
///
/// Per chunk (≤126 phonemes, ≤384 frames ≈ 9.6 s):
/// acoustic (text encoder, duration predictor, prosody, decoder; CPU, uses
/// the `bidirectional_lstm` custom op) → vocoder (GPU when the runtime can
/// compile it, else CPU) → host iSTFT. English only: text is phonemized on
/// the host with the CMU dictionary plus a LiteRT neural G2P for unknown
/// words. The graphs have no speed input, so the speaking rate is applied
/// to the audio with a pitch-preserving time stretch.
///
/// Style: a voice row is 256 floats; the acoustic graph (duration predictor)
/// takes the second half and the vocoder (decoder) the first, as in the
/// reference model.
class KokoroEngine extends TtsEngine {
  KokoroEngine._(
    this._acoustic,
    this._vocoder,
    this._g2pGraph,
    this._vocab,
    this._cmu,
    this._voiceFiles,
    this._vocoderNote,
  ) {
    _ac = _acoustic.sig();
    _voc = _vocoder.sig();
    _tokens = _ac.input('phoneme_ids').shape.last;
    _frames = _ac.output('acoustic_features').shape.last;
    _g2p = _g2pGraph?.sig();
  }

  static const acousticFile = 'acoustic.tflite';
  static const vocoderFile = 'vocoder.tflite';
  static const g2pFile = 'dp_g2p_litert.tflite';
  static const configFile = 'config.json';
  static const cmuFile = 'cmudict.txt';
  static const voicesDir = 'voices';

  static const englishVoices = KokoroVoices.english;

  /// Floats per voice row; the first [halfDim] drive the decoder, the rest
  /// the duration predictor.
  static const styleDim = 256;
  static const halfDim = styleDim ~/ 2;

  final LiteRtGraph _acoustic, _vocoder;
  final LiteRtGraph? _g2pGraph;
  final Map<String, int> _vocab;
  final Map<String, String> _cmu;
  final Map<String, String> _voiceFiles;

  /// Why the vocoder runs on the CPU although a GPU was requested.
  final String? _vocoderNote;
  final Map<String, Float32List> _styles = {};

  late final Sig _ac, _voc;
  late final Sig? _g2p;
  late final int _tokens, _frames;

  static KokoroEngine load(String dir, InferenceOptions o) {
    String f(String rel) => p.joinAll([dir, ...rel.split('/')]);
    final graphs = <LiteRtGraph>[];
    try {
      // The acoustic graph stays on the CPU: it holds the custom LSTM op,
      // which only has a CPU kernel.
      final acoustic = LiteRtGraph.open(f(acousticFile), o);
      graphs.add(acoustic);
      LiteRtGraph vocoder;
      String? note;
      try {
        vocoder = LiteRtGraph.open(f(vocoderFile), o, accelerate: true);
      } on Object catch (e) {
        // Some GPU runtimes (e.g. WebGPU in the current LiteRT) fail to
        // compile the vocoder; run it on the CPU instead.
        vocoder = LiteRtGraph.open(
          f(vocoderFile),
          o.copyWith(accelerator: AcceleratorKind.cpu),
        );
        note = 'GPU unavailable for the vocoder: ${e.toString().split('\n').first}';
      }
      graphs.add(vocoder);
      LiteRtGraph? g2p;
      if (File(f(g2pFile)).existsSync()) {
        // Tiny graph: one thread avoids pool overhead.
        g2p = LiteRtGraph.open(f(g2pFile), o, threads: 1);
        graphs.add(g2p);
      }
      final voices = <String, String>{};
      final vd = Directory(f(voicesDir));
      if (vd.existsSync()) {
        for (final e in vd.listSync().whereType<File>()) {
          if (e.path.endsWith('.bin')) {
            voices[p.basenameWithoutExtension(e.path)] = e.path;
          }
        }
      }
      if (voices.isEmpty) throw StateError('No Kokoro voices in ${vd.path}');
      return KokoroEngine._(
        acoustic,
        vocoder,
        g2p,
        EnglishPhonemizer.loadVocab(f(configFile)),
        EnglishPhonemizer.loadCmu(f(cmuFile)),
        voices,
        note,
      );
    } catch (_) {
      for (final g in graphs) {
        g.close();
      }
      rethrow;
    }
  }

  @override
  TtsEngineKind get kind => TtsEngineKind.kokoro;

  @override
  String get accelerator => _vocoderNote == null
      ? 'acoustic CPU · vocoder ${_vocoder.accelerator}'
      : 'CPU (GPU unavailable for the vocoder)';

  @override
  String? get fallbackReason => _vocoderNote;

  @override
  List<TtsVoice> get voices {
    final ids = _voiceFiles.keys.toList()
      ..sort((a, b) {
        final ia = englishVoices.indexOf(a), ib = englishVoices.indexOf(b);
        return (ia < 0 ? 999 : ia).compareTo(ib < 0 ? 999 : ib);
      });
    return [for (final id in ids) voiceInfo(id)];
  }

  static TtsVoice voiceInfo(String id) {
    final accent = switch (id[0]) {
      'a' => 'American English',
      'b' => 'British English',
      _ => 'Other',
    };
    final gender = id.length > 1 && id[1] == 'm' ? 'Male' : 'Female';
    final name = id.contains('_') ? id.substring(id.indexOf('_') + 1) : id;
    final label = name.isEmpty ? id : '${name[0].toUpperCase()}${name.substring(1)}';
    return TtsVoice(id, label, group: '$accent · $gender');
  }

  @override
  void close() {
    _acoustic.close();
    _vocoder.close();
    _g2pGraph?.close();
  }

  @override
  Future<void> warmUp(WorkerContext ctx) async {
    final ids = List.filled(_tokens, 0);
    ids[1] = _vocab['a']!;
    _ac.setInt64('phoneme_ids', ids);
    _ac.setInt64('phoneme_length', [3]);
    _ac.invoke();
    await ctx.checkpoint();
    _voc.input('speech_frame_length').asInt32View()[0] = 1;
    _voc.invoke();
    _g2p?.invoke();
  }

  Float32List _style(String voice, int phonemes) {
    final table = _styles[voice] ??= () {
      final path = _voiceFiles[voice] ??
          (throw ArgumentError('Unknown Kokoro voice "$voice"'));
      final bytes = File(path).readAsBytesSync();
      return Float32List.fromList(bytes.buffer.asFloat32List(0, bytes.length ~/ 4));
    }();
    final rows = table.length ~/ styleDim;
    final i = (phonemes - 1).clamp(0, rows - 1);
    return Float32List.sublistView(table, i * styleDim, (i + 1) * styleDim);
  }

  // ---- Neural G2P (DeepPhonemizer en_us_cmudict_forward) -----------------

  static const _g2pLen = 96;
  static const _g2pPhonemes = 42;
  static const _arpa = [
    '', '', '', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'DH', 'D', //
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'NG',
    'N', 'OW', 'OY', 'P', 'R', 'SH', 'S', 'TH', 'T', 'UH', 'UW', 'V', 'W',
    'Y', 'ZH', 'Z',
  ];

  List<String> _neuralG2p(String word) {
    final g = _g2p;
    if (g == null) return const [];
    // [LANG=1] + each char id ×3 + [END=2], zero padded.
    final ids = <int>[1];
    for (final c in word.codeUnits) {
      final id = c == 0x27 ? 3 : (c >= 0x61 && c <= 0x7a ? c - 0x61 + 4 : -1);
      if (id >= 0) ids.addAll([id, id, id]);
    }
    ids.add(2);
    if (ids.length <= 2) return const [];
    final len = math.min(ids.length, _g2pLen);
    final input = g.inF(g.inputNames.first)..fillRange(0, _g2pLen, 0);
    for (var i = 0; i < len; i++) {
      input[i] = ids[i].toDouble();
    }
    g.invoke();
    final logits = g.outF(g.outputNames.first);
    final out = <String>[];
    var prev = -1;
    for (var t = 0; t < len; t++) {
      var best = 0;
      var bestV = logits[t * _g2pPhonemes];
      for (var k = 1; k < _g2pPhonemes; k++) {
        final v = logits[t * _g2pPhonemes + k];
        if (v > bestV) {
          bestV = v;
          best = k;
        }
      }
      if (best == prev) continue;
      prev = best;
      if (best <= 2) continue; // pad / lang / end
      out.add(_arpa[best]);
    }
    return out;
  }

  // ---- Chunking -------------------------------------------------------------

  /// Packs phonemized sentences into chunks of at most `bucket - 2` ids
  /// (BOS/EOS take two slots).
  List<List<int>> _chunks(String text, EnglishPhonemizer g2p) {
    final max = _tokens - 2;
    final space = g2p.spaceId;
    final pieces = <List<int>>[];
    for (final s in splitSentences(text)) {
      var ids = g2p.phonemize(s);
      while (ids.length > max) {
        // Prefer breaking after punctuation, else at a word boundary.
        var cut = -1;
        for (var i = max - 1; i > max ~/ 2; i--) {
          if (ids[i] >= 1 && ids[i] <= 6) {
            cut = i + 1;
            break;
          }
        }
        if (cut < 0) cut = ids.lastIndexOf(space, max - 1);
        if (cut <= 0) cut = max;
        pieces.add(ids.sublist(0, cut));
        ids = ids.sublist(cut).skipWhile((i) => i == space).toList();
      }
      if (ids.isNotEmpty) pieces.add(ids);
    }
    final out = <List<int>>[];
    var cur = <int>[];
    for (final piece in pieces) {
      if (cur.isNotEmpty && cur.length + 1 + piece.length <= max) {
        cur = [...cur, space, ...piece];
      } else {
        if (cur.isNotEmpty) out.add(cur);
        cur = piece;
      }
    }
    if (cur.isNotEmpty) out.add(cur);
    return out;
  }

  // ---- Synthesis -------------------------------------------------------------

  @override
  Future<TtsStats> synthesize(
    TtsRequest req,
    WorkerContext ctx,
    void Function(Float32List pcm) emit,
  ) async {
    final g2p = EnglishPhonemizer(
      _cmu,
      _vocab,
      fallback: _g2p == null ? null : _neuralG2p,
      british: req.voiceId.startsWith('b'),
    );
    final speed = req.speed.clamp(0.5, 2.0).toDouble();
    final total = Stopwatch()..start();
    double? firstAudio;
    var samples = 0, chunkCount = 0;
    final stages = <String, double>{
      'g2p': 0,
      'acoustic': 0,
      'vocoder': 0,
      'istft': 0,
      'stretch': 0,
    };
    final sw = Stopwatch()..start();
    double lap() {
      final ms = sw.elapsedMicroseconds / 1000;
      sw
        ..reset()
        ..start();
      return ms;
    }

    final queue = _chunks(req.text, g2p);
    stages['g2p'] = lap();
    while (queue.isNotEmpty) {
      final tokens = queue.removeAt(0);
      var pcm = _runChunk(tokens, req.voiceId, stages, lap);
      if (pcm == null) {
        // Too many frames for the graph: split at a word boundary.
        final mid = tokens.lastIndexOf(_vocab[' ']!, tokens.length ~/ 2 + 8);
        final cut = mid > 0 ? mid : tokens.length ~/ 2;
        queue.insertAll(0, [tokens.sublist(0, cut), tokens.sublist(cut + (mid > 0 ? 1 : 0))]);
        continue;
      }
      if ((speed - 1).abs() > 1e-3) {
        pcm = timeStretch(pcm, speed, sampleRate: sampleRate);
        stages['stretch'] = stages['stretch']! + lap();
      }
      firstAudio ??= total.elapsedMicroseconds / 1000;
      samples += pcm.length;
      chunkCount++;
      emit(pcm);
      await ctx.checkpoint();
    }
    return TtsStats(
      samples: samples,
      sampleRate: sampleRate,
      totalMs: total.elapsedMicroseconds / 1000,
      firstAudioMs: firstAudio ?? 0,
      units: chunkCount,
      unitLabel: 'chunk',
      stagesMs: stages,
    );
  }

  /// Runs one chunk; returns null if its frames exceed the graph's limit.
  Float32List? _runChunk(
    List<int> tokens,
    String voice,
    Map<String, double> stages,
    double Function() lap,
  ) {
    final style = _style(voice, tokens.length);
    final ids = List.filled(_tokens, 0)..setAll(1, tokens);

    lap();
    _ac.setInt64('phoneme_ids', ids);
    _ac.setInt64('phoneme_length', [tokens.length + 2]);
    _ac.inF('speaker_style').setAll(0, Float32List.sublistView(style, halfDim));
    _ac.invoke();
    var frames = _ac.output('speech_frame_length').asInt64View()[0];
    stages['acoustic'] = stages['acoustic']! + lap();
    if (frames >= _frames) {
      // The graph's frame budget is exhausted (it clamps the length).
      if (tokens.length >= 4) return null;
      frames = _frames; // Can't split further; keep what fits.
    }
    if (frames <= 0) return Float32List(0);

    _voc.inF('acoustic_features').setAll(0, _ac.outF('acoustic_features'));
    _voc.inF('energy_contour').setAll(0, _ac.outF('energy_contour'));
    _voc.inF('pitch_contour').setAll(0, _ac.outF('pitch_contour'));
    _voc.inF('speaker_style').setAll(0, Float32List.sublistView(style, 0, halfDim));
    _voc.input('speech_frame_length').asInt32View()[0] = frames;
    _voc.invoke();
    stages['vocoder'] = stages['vocoder']! + lap();

    final mag = _voc.outF('magnitude_spectrogram');
    final phase = _voc.outF('phase_spectrogram');
    final stftFrames = mag.length ~/ KokoroDsp.bins;
    final pcm = KokoroDsp.istft(mag, phase, stftFrames, frames * KokoroDsp.samplesPerFrame);
    stages['istft'] = stages['istft']! + lap();
    return pcm;
  }
}
