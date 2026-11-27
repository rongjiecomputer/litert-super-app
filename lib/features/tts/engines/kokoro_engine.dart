import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import '../../../core/inference/inference_options.dart';
import '../../../core/inference/inference_worker.dart';
import '../../../core/models/model_catalog.dart' show KokoroVoices;
import '../dsp/kokoro_dsp.dart';
import '../text/english_g2p.dart';
import 'graph.dart';
import 'tts_engine.dart';

/// Kokoro-82M (StyleTTS2 + iSTFTNet) on the bucketed LiteRT graphs.
///
/// Per chunk (≤126 phonemes, ≤512 frames ≈ 12.8 s):
/// predictor → host duration rounding + alignment → prosody → host
/// harmonic source + STFT → vocoder → host iSTFT. English only: text is
/// phonemized on the host with the CMU dictionary plus a LiteRT neural G2P
/// for unknown words.
class KokoroEngine extends TtsEngine {
  KokoroEngine._(
    this._predictor,
    this._prosody,
    this._vocoder,
    this._g2pGraph,
    this._vocab,
    this._cmu,
    this._voiceFiles,
  ) {
    _pred = _predictor.sig();
    _pros = _prosody.sig();
    _voc = _vocoder.sig();
    _tokens = _pred.input('args_0').shape.last;
    _frames = _pros.input('args_4').shape.last;
    _g2p = _g2pGraph?.sig();
  }

  static const predictorFile = 'kokoro_predictor.tflite';
  static const prosodyFile = 'kokoro_prosody.tflite';
  static const vocoderFile = 'kokoro_vocoder.tflite';
  static const g2pFile = 'dp_g2p_litert.tflite';
  static const configFile = 'config.json';
  static const cmuFile = 'cmudict.txt';
  static const voicesDir = 'voices';

  static const englishVoices = KokoroVoices.english;

  static const styleDim = 256;

  final LiteRtGraph _predictor, _prosody, _vocoder;
  final LiteRtGraph? _g2pGraph;
  final Map<String, int> _vocab;
  final Map<String, String> _cmu;
  final Map<String, String> _voiceFiles;
  final _dsp = KokoroDsp.kokoro82m();
  final Map<String, Float32List> _styles = {};

  late final Sig _pred, _pros, _voc;
  late final Sig? _g2p;
  late final int _tokens, _frames;

  static KokoroEngine load(String dir, InferenceOptions o) {
    String f(String rel) => p.joinAll([dir, ...rel.split('/')]);
    final graphs = <LiteRtGraph>[];
    try {
      final pred = LiteRtGraph.open(f(predictorFile), o);
      graphs.add(pred);
      final pros = LiteRtGraph.open(f(prosodyFile), o);
      graphs.add(pros);
      final voc = LiteRtGraph.open(f(vocoderFile), o);
      graphs.add(voc);
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
        pred,
        pros,
        voc,
        g2p,
        EnglishPhonemizer.loadVocab(f(configFile)),
        EnglishPhonemizer.loadCmu(f(cmuFile)),
        voices,
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
  String get accelerator => _vocoder.accelerator;

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
    _predictor.close();
    _prosody.close();
    _vocoder.close();
    _g2pGraph?.close();
  }

  @override
  Future<void> warmUp(WorkerContext ctx) async {
    final ids = List.filled(_tokens, 0);
    ids[1] = _vocab['a']!;
    _pred.setInt64('args_0', ids);
    _pred.inF('args_2')
      ..fillRange(0, _tokens, 0)
      ..fillRange(0, 3, 1);
    _pred.invoke();
    await ctx.checkpoint();
    _pros.invoke();
    await ctx.checkpoint();
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
    final rng = math.Random(req.seed);
    final total = Stopwatch()..start();
    double? firstAudio;
    var samples = 0, chunkCount = 0;
    final stages = <String, double>{'g2p': 0, 'predictor': 0, 'prosody': 0, 'source': 0, 'vocoder': 0, 'istft': 0};
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
      final pcm = _runChunk(tokens, req, rng, stages, lap);
      if (pcm == null) {
        // Too many frames for the bucket: split at a word boundary.
        final mid = tokens.lastIndexOf(_vocab[' ']!, tokens.length ~/ 2 + 8);
        final cut = mid > 0 ? mid : tokens.length ~/ 2;
        queue.insertAll(0, [tokens.sublist(0, cut), tokens.sublist(cut + (mid > 0 ? 1 : 0))]);
        continue;
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

  /// Runs one chunk; returns null if its frames exceed the bucket.
  Float32List? _runChunk(
    List<int> tokens,
    TtsRequest req,
    math.Random rng,
    Map<String, double> stages,
    double Function() lap,
  ) {
    final t = tokens.length + 2;
    final style = _style(req.voiceId, tokens.length);
    final ids = List.filled(_tokens, 0)..setAll(1, tokens);

    lap();
    _pred.setInt64('args_0', ids);
    _pred.inF('args_1').setAll(0, style);
    _pred.inF('args_2')
      ..fillRange(0, _tokens, 0)
      ..fillRange(0, t, 1);
    _pred.invoke();
    final dur = _pred.outF('output_0');
    final speed = req.speed.clamp(0.5, 2.0);
    final durations = [
      for (var i = 0; i < t; i++) math.max(1, (dur[i] / speed).round()),
    ];
    final frames = durations.fold(0, (a, b) => a + b);
    stages['predictor'] = stages['predictor']! + lap();
    if (frames > _frames) {
      if (tokens.length < 4) {
        // Can't split further; truncate the alignment.
        return _finish(durations, _frames, style, rng, stages, lap);
      }
      return null;
    }
    return _finish(durations, frames, style, rng, stages, lap);
  }

  Float32List _finish(
    List<int> durations,
    int frames,
    Float32List style,
    math.Random rng,
    Map<String, double> stages,
    double Function() lap,
  ) {
    _pros.inF('args_0').setAll(0, _pred.outF('output_1'));
    _pros.inF('args_1').setAll(0, _pred.outF('output_2'));
    final aln = _pros.inF('args_2')..fillRange(0, _tokens * _frames, 0);
    var f = 0;
    for (var i = 0; i < durations.length && f < _frames; i++) {
      for (var k = 0; k < durations[i] && f < _frames; k++) {
        aln[i * _frames + f++] = 1;
      }
    }
    _pros.inF('args_3').setAll(0, style);
    _pros.inF('args_4')
      ..fillRange(0, _frames, 0)
      ..fillRange(0, frames, 1);
    _pros.invoke();
    stages['prosody'] = stages['prosody']! + lap();

    final f0 = Float32List.fromList(_pros.outF('output_1'));
    final har = _dsp.harmonicSource(f0, rng);
    stages['source'] = stages['source']! + lap();

    _voc.inF('args_0').setAll(0, _pros.outF('output_0'));
    _voc.inF('args_1').setAll(0, f0);
    _voc.inF('args_2').setAll(0, _pros.outF('output_2'));
    _voc.inF('args_3').setAll(0, har);
    _voc.inF('args_4').setAll(0, style);
    _voc.inF('args_5')
      ..fillRange(0, _frames, 0)
      ..fillRange(0, frames, 1);
    _voc.invoke();
    stages['vocoder'] = stages['vocoder']! + lap();

    final mag = _voc.outF('output_0'), phase = _voc.outF('output_1');
    final stftFrames = mag.length ~/ KokoroDsp.bins;
    final pcm = KokoroDsp.istft(mag, phase, stftFrames, frames * 600);
    stages['istft'] = stages['istft']! + lap();
    return pcm;
  }
}
