import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import '../../../core/inference/inference_options.dart';
import '../../../core/inference/inference_worker.dart';
import '../text/english_g2p.dart' show splitSentences;
import '../text/npy.dart';
import '../text/qwen_tokenizer.dart';
import 'graph.dart';
import 'sampling.dart';
import 'tts_engine.dart';

/// Qwen3-TTS-12Hz-0.6B-Base on LiteRT, host-orchestrated.
///
/// A Dart port of `qwen3_tts_pipeline.py` (litert-samples, Apache-2.0): a
/// 28-layer talker predicts codebook 0 at 12.5 Hz, a 5-layer MTP predicts
/// the 15 residual codebooks per frame, and the codec decoder turns 16×T
/// codes into 24 kHz PCM. Voices are 1024-d speaker x-vectors (the bundled
/// demo voice, or one extracted by [SpeakerEncoder] for cloning).
class Qwen3TtsEngine extends TtsEngine {
  Qwen3TtsEngine._(
    this._tok,
    this._codecEmb,
    this._mtpEmb,
    this._textEmb,
    this._proj,
    this._demoSpeaker,
    this._talker,
    this._mtp,
    this._codec,
  ) {
    final ids = [ttsBos, ttsEos, ttsPad];
    final rows = _embedText(ids);
    _ttsBos = rows[0];
    _ttsEos = rows[1];
    _ttsPad = rows[2];
    _prefill = _talker.sig('prefill_32');
    _decode = _talker.sig('decode');
    _kvNames = [
      for (final n in _decode.inputNames)
        if (n.startsWith('kv_cache')) n,
    ];
    _cacheLen = _decode.input('mask').shape.last;
    _mtpSig = _mtp.sig();
    _mtpCache = _mtpSig.input('args_2').shape.last;
    _codecSig = _codec.sig();
    _codecChunk = _codecSig.input('args_0').shape.last;
  }

  // Model files (relative to the model folder).
  static const talkerFile = 'talker_int4.tflite';
  static const mtpFile = 'mtp_fp32.tflite';
  static const codecFile = 'codec_decoder_fp32.tflite';
  static const tokenizerFile = 'tokenizer.json';
  static const codecEmbFile = 'tables/codec_embedding_fp32.npy';
  static const mtpEmbFile = 'tables/mtp_embeddings_fp16.npy';
  static const textEmbFile = 'tables/text_embedding_fp16.npy';
  static const projFile = 'tables/text_projection_fp32.npz';
  static const demoSpeakerFile = 'voices/demo_speaker.npy';
  static const speakerEncoderFile = 'speaker_encoder.tflite';

  // Talker codec vocabulary (ids >= 2048 are control tokens).
  static const codecVocab = 3072;
  static const codecPad = 2148;
  static const codecBos = 2149;
  static const codecEos = 2150;
  static const codecThink = 2154;
  static const codecNoThink = 2155;
  static const codecThinkBos = 2156;
  static const codecThinkEos = 2157;

  // Text-side special tokens.
  static const ttsPad = 151671;
  static const ttsBos = 151672;
  static const ttsEos = 151673;

  static const hidden = 1024;
  static const codeGroups = 16;
  static const samplesPerFrame = 1920; // 24000 / 12.5
  static const leftContext = 25;
  static const negInf = -1e9;

  /// Codec language ids (talker_config.codec_language_id).
  static const languages = {
    'english': 2050,
    'german': 2053,
    'spanish': 2054,
    'chinese': 2055,
    'japanese': 2058,
    'french': 2061,
    'korean': 2064,
    'russian': 2069,
    'italian': 2070,
    'portuguese': 2071,
  };

  static const demoVoice = TtsVoice('demo', 'Demo speaker', group: 'Bundled');

  final QwenTokenizer _tok;
  final NpyArray _codecEmb; // [3072, 1024]
  final NpyArray _mtpEmb; // [15, 2048, 1024]
  final NpyRowReader _textEmb; // [151936, 2048] fp16, read per row
  final Map<String, NpyArray> _proj; // w1 [2048,2048] b1 w2 [1024,2048] b2
  final Float32List _demoSpeaker;
  final LiteRtGraph _talker, _mtp, _codec;

  late final Float32List _ttsBos, _ttsEos, _ttsPad;
  late final Sig _prefill, _decode, _mtpSig, _codecSig;
  late final List<String> _kvNames;
  late final int _cacheLen, _mtpCache, _codecChunk;

  /// Frames decoded before the first audio chunk is emitted (lower =
  /// earlier playback, at the cost of one extra codec call).
  int firstWindowFrames = 16;

  static Qwen3TtsEngine load(String dir, InferenceOptions o) {
    String f(String rel) => p.joinAll([dir, ...rel.split('/')]);
    final graphs = <LiteRtGraph>[];
    try {
      final talker = LiteRtGraph.open(f(talkerFile), o);
      graphs.add(talker);
      // The MTP's matmuls are small; a few threads is the sweet spot.
      final mtp = LiteRtGraph.open(f(mtpFile), o, threads: math.min(4, o.cpuThreads));
      graphs.add(mtp);
      // The codec decoder is the graph that benefits from the GPU.
      final codec = LiteRtGraph.open(f(codecFile), o, accelerate: true);
      graphs.add(codec);
      return Qwen3TtsEngine._(
        QwenTokenizer.fromFile(f(tokenizerFile)),
        NpyArray.readFile(f(codecEmbFile)),
        NpyArray.readFile(f(mtpEmbFile)),
        NpyRowReader.open(f(textEmbFile)),
        NpyArray.readNpz(f(projFile)),
        NpyArray.readFile(f(demoSpeakerFile)).data,
        talker,
        mtp,
        codec,
      );
    } catch (_) {
      for (final g in graphs) {
        g.close();
      }
      rethrow;
    }
  }

  @override
  TtsEngineKind get kind => TtsEngineKind.qwen3;

  @override
  List<TtsVoice> get voices => const [demoVoice];

  @override
  String get accelerator =>
      'talker ${_talker.accelerator} · codec ${_codec.accelerator}';

  @override
  void close() {
    _talker.close();
    _mtp.close();
    _codec.close();
    _textEmb.close();
  }

  @override
  Future<void> warmUp(WorkerContext ctx) async {
    for (final n in _kvNames) {
      _decode.inF(n).fillRange(0, _decode.inF(n).length, 0);
    }
    _decode.inF('embeddings').fillRange(0, hidden, 0);
    _decode.setInt32('input_pos', [0]);
    _decode.inF('mask').fillRange(0, _cacheLen, 0);
    _decode.invoke();
    await ctx.checkpoint();
    _mtpSig.inF('args_3').fillRange(0, _mtpSig.inF('args_3').length, 0);
    _mtpSig.inF('args_4').fillRange(0, _mtpSig.inF('args_4').length, 0);
    _mtpSig.setInt32('args_1', [0]);
    _mtpSig.invoke();
    await ctx.checkpoint();
    _codecSig.setInt32('args_0', List.filled(codeGroups * _codecChunk, 0));
    _codecSig.invoke();
  }

  // ---- Text conditioning ---------------------------------------------------

  /// Looks up text-embedding rows and applies the 2048→1024 projection MLP.
  List<Float32List> _embedText(List<int> ids) {
    final w1 = _proj['w1']!.data, b1 = _proj['b1']!.data;
    final w2 = _proj['w2']!.data, b2 = _proj['b2']!.data;
    final inDim = _proj['w1']!.shape[1], mid = _proj['w1']!.shape[0];
    final outDim = _proj['w2']!.shape[0];
    final h = Float32List(mid);
    return [
      for (final id in ids)
        () {
          final x = _textEmb.row(id);
          for (var o = 0; o < mid; o++) {
            var s = b1[o];
            final base = o * inDim;
            for (var i = 0; i < inDim; i++) {
              s += w1[base + i] * x[i];
            }
            h[o] = s / (1 + math.exp(-s)); // SiLU
          }
          final y = Float32List(outDim);
          for (var o = 0; o < outDim; o++) {
            var s = b2[o];
            final base = o * mid;
            for (var i = 0; i < mid; i++) {
              s += w2[base + i] * h[i];
            }
            y[o] = s;
          }
          return y;
        }(),
    ];
  }

  /// Embedding of [code] in residual codebook table [k].
  Float32List _mtpRow(int k, int code) {
    final start = (k * _mtpEmb.shape[1] + code) * hidden;
    return Float32List.sublistView(_mtpEmb.data, start, start + hidden);
  }

  Float32List _add(Float32List a, Float32List b) {
    final out = Float32List(hidden);
    for (var i = 0; i < hidden; i++) {
      out[i] = a[i] + b[i];
    }
    return out;
  }

  /// Prefill rows `[P][1024]`, streamed text conditioning `[L][1024]`.
  (List<Float32List>, List<Float32List>) _buildPrompt(
    String text,
    Float32List speaker,
    String language,
  ) {
    final ids = _tok.encode(
      '<|im_start|>assistant\n$text<|im_end|>\n<|im_start|>assistant\n',
    );
    final lang = languages[language.toLowerCase()];
    final control = lang == null
        ? [codecNoThink, codecThinkBos, codecThinkEos]
        : [codecThink, codecThinkBos, lang, codecThinkEos];
    final codecPre = <Float32List>[
      for (final c in control) _codecEmb.row(c),
      speaker,
      _codecEmb.row(codecPad),
      _codecEmb.row(codecBos),
    ];
    final role = _embedText(ids.sublist(0, 3));
    final body = <Float32List>[
      for (var j = 0; j < codecPre.length - 1; j++)
        _add(j < codecPre.length - 2 ? _ttsPad : _ttsBos, codecPre[j]),
    ];
    final firstText = _add(_embedText([ids[3]]).first, codecPre.last);
    final trailingIds = ids.length > 9 ? ids.sublist(4, ids.length - 5) : <int>[];
    final trailing = [..._embedText(trailingIds), _ttsEos];
    return ([...role, ...body, firstText], trailing);
  }

  // ---- Graph steps ---------------------------------------------------------

  void _runPrefill(List<Float32List> rows) {
    const n = 32;
    final p = rows.length;
    if (p > n) throw StateError('Prompt too long for prefill_32: $p');
    final emb = _prefill.inF('embeddings')..fillRange(0, n * hidden, 0);
    for (var i = 0; i < p; i++) {
      emb.setAll(i * hidden, rows[i]);
    }
    _prefill.setInt32('input_pos', List.generate(n, (i) => i));
    final mask = _prefill.inF('mask');
    for (var i = 0; i < n; i++) {
      final visible = math.min(i, p - 1) + 1;
      mask.fillRange(i * _cacheLen, i * _cacheLen + visible, 0);
      mask.fillRange(i * _cacheLen + visible, (i + 1) * _cacheLen, negInf);
    }
    for (final k in _kvNames) {
      final v = _prefill.inF(k);
      v.fillRange(0, v.length, 0);
    }
    _prefill.invoke();
    for (final k in _kvNames) {
      _decode.inF(k).setAll(0, _prefill.outF(k));
    }
    _decode.inF('mask').fillRange(0, _cacheLen, negInf);
  }

  final Float32List _logits = Float32List(codecVocab);
  final Float32List _hidden = Float32List(hidden);

  /// One talker decode step; fills [_logits] (cb0) and [_hidden].
  void _runDecode(Float32List embed, int pos) {
    _decode.inF('embeddings').setAll(0, embed);
    _decode.setInt32('input_pos', [pos]);
    _decode.inF('mask').fillRange(0, pos + 1, 0);
    _decode.invoke();
    for (final k in _kvNames) {
      _decode.inF(k).setAll(0, _decode.outF(k));
    }
    final out = _decode.outF('logits');
    _logits.setAll(0, Float32List.sublistView(out, 0, codecVocab));
    _hidden.setAll(0, Float32List.sublistView(out, codecVocab, codecVocab + hidden));
  }

  /// 16-step MTP inner loop for one frame → 15 residual codes.
  List<int> _runMtp(int cb0, math.Random rng, TtsRequest r) {
    final k = _mtpSig.inF('args_3'), v = _mtpSig.inF('args_4');
    k.fillRange(0, k.length, 0);
    v.fillRange(0, v.length, 0);
    final codes = <int>[];
    final mask = _mtpSig.inF('args_2');
    final emb = _mtpSig.inF('args_0');
    for (var t = 0; t < codeGroups; t++) {
      if (t == 0) {
        emb.setAll(0, _hidden);
      } else if (t == 1) {
        emb.setAll(0, _codecEmb.row(cb0));
      } else {
        emb.setAll(0, _mtpRow(t - 2, codes.last));
      }
      _mtpSig.setInt32('args_1', [t]);
      for (var i = 0; i < _mtpCache; i++) {
        mask[i] = i <= t ? 0 : negInf;
      }
      _mtpSig.invoke();
      k.setAll(0, _mtpSig.outF('output_1'));
      v.setAll(0, _mtpSig.outF('output_2'));
      if (t >= 1) {
        final heads = _mtpSig.outF('output_0');
        final vocab = heads.length ~/ (codeGroups - 1);
        codes.add(
          pickToken(
            heads,
            rng,
            greedy: r.greedy,
            topK: r.topK,
            temperature: r.temperature,
            offset: (t - 1) * vocab,
            length: vocab,
          ),
        );
      }
    }
    return codes;
  }

  /// Decodes `frames[start-c .. end)` with the fixed 64-frame codec graph;
  /// returns the PCM for frames `[start, end)`.
  Float32List _decodeWindow(List<Int32List> frames, int start, int end) {
    final c = math.min(leftContext, start);
    final from = start - c;
    final len = end - from;
    final buf = Int32List(codeGroups * _codecChunk);
    for (var t = 0; t < len; t++) {
      final f = frames[from + t];
      for (var g = 0; g < codeGroups; g++) {
        buf[g * _codecChunk + t] = f[g];
      }
    }
    _codecSig.input('args_0').setTo(buf);
    _codecSig.invoke();
    final wav = _codecSig.outF(_codecSig.outputNames.first);
    return Float32List.fromList(
      Float32List.sublistView(wav, c * samplesPerFrame, len * samplesPerFrame),
    );
  }

  // ---- Synthesis -----------------------------------------------------------

  @override
  Future<TtsStats> synthesize(
    TtsRequest req,
    WorkerContext ctx,
    void Function(Float32List pcm) emit,
  ) async {
    final speaker = req.speakerEmbedding ?? _demoSpeaker;
    if (speaker.length != hidden) {
      throw ArgumentError('Speaker embedding must have $hidden values');
    }
    final rng = math.Random(req.seed);
    final total = Stopwatch()..start();
    double? firstAudio;
    var samples = 0, frameCount = 0;
    final stages = <String, double>{'prefill': 0, 'talker': 0, 'mtp': 0, 'codec': 0};
    final sw = Stopwatch();
    double lap() {
      final ms = sw.elapsedMicroseconds / 1000;
      sw
        ..reset()
        ..start();
      return ms;
    }

    void out(Float32List pcm) {
      if (pcm.isEmpty) return;
      firstAudio ??= total.elapsedMicroseconds / 1000;
      samples += pcm.length;
      emit(pcm);
    }

    final chunks = _groupForSynthesis(req.text);
    for (var ci = 0; ci < chunks.length; ci++) {
      final text = chunks[ci];
      sw
        ..reset()
        ..start();
      final (prefill, trailing) = _buildPrompt(text, speaker, req.language);
      _runPrefill(prefill);
      var pos = prefill.length - 1;
      _runDecode(prefill.last, pos);
      stages['prefill'] = stages['prefill']! + lap();
      await ctx.checkpoint();

      final frames = <Int32List>[];
      final history = <int>{};
      var emitted = 0;
      final maxFrames = _frameBudget(text);
      Future<void> flush({required bool last}) async {
        while (true) {
          final avail = frames.length - emitted;
          if (avail <= 0) return;
          final c = math.min(leftContext, emitted);
          final cap = _codecChunk - c;
          final need = emitted == 0 ? math.min(firstWindowFrames, cap) : cap;
          if (!last && avail < need) return;
          final end = math.min(emitted + cap, frames.length);
          lap();
          out(_decodeWindow(frames, emitted, end));
          stages['codec'] = stages['codec']! + lap();
          emitted = end;
          await ctx.checkpoint();
        }
      }

      while (frames.length < maxFrames) {
        final scores = Float32List.fromList(_logits);
        for (var i = 2048; i < codecVocab; i++) {
          if (i != codecEos) scores[i] = negInf;
        }
        if (frames.length < 2) scores[codecEos] = negInf; // min 2 frames
        applyRepetitionPenalty(scores, history, 1.05);
        final cb0 = pickToken(
          scores,
          rng,
          greedy: req.greedy,
          topK: req.topK,
          temperature: req.temperature,
        );
        history.add(cb0);
        if (cb0 == codecEos) break;

        lap();
        final residual = _runMtp(cb0, rng, req);
        stages['mtp'] = stages['mtp']! + lap();
        final frame = Int32List(codeGroups)
          ..[0] = cb0
          ..setAll(1, residual);
        frames.add(frame);

        final embed = Float32List.fromList(_codecEmb.row(cb0));
        for (var k = 0; k < codeGroups - 1; k++) {
          final e = _mtpRow(k, residual[k]);
          for (var i = 0; i < hidden; i++) {
            embed[i] += e[i];
          }
        }
        final step = frames.length - 1;
        final cond = step < trailing.length ? trailing[step] : _ttsPad;
        for (var i = 0; i < hidden; i++) {
          embed[i] += cond[i];
        }
        pos++;
        if (pos >= _cacheLen) break;
        _runDecode(embed, pos);
        stages['talker'] = stages['talker']! + lap();
        await flush(last: false);
        await ctx.checkpoint();
      }
      await flush(last: true);
      frameCount += frames.length;
      if (ci < chunks.length - 1) out(Float32List(sampleRate * 15 ~/ 100));
    }
    return TtsStats(
      samples: samples,
      sampleRate: sampleRate,
      totalMs: total.elapsedMicroseconds / 1000,
      firstAudioMs: firstAudio ?? 0,
      units: frameCount,
      unitLabel: 'frame',
      stagesMs: stages,
    );
  }

  /// Rough spoken duration: CJK ≈ 5 chars/s, other scripts ≈ 14 chars/s.
  static double estimateSeconds(String s) {
    var cjk = 0, other = 0;
    for (final r in s.runes) {
      if ((r >= 0x3040 && r <= 0x30ff) ||
          (r >= 0x3400 && r <= 0x9fff) ||
          (r >= 0xac00 && r <= 0xd7af)) {
        cjk++;
      } else if (r > 0x20) {
        other++;
      }
    }
    return cjk / 5 + other / 14;
  }

  /// Joins sentences into chunks of ≤ ~20 s of speech each (one talker run
  /// per chunk; the KV cache and frame cap bound a single run to ~41 s).
  static List<String> _groupForSynthesis(String text) {
    final sentences = splitSentences(text.trim(), maxChars: 200);
    final out = <String>[];
    var cur = '';
    for (final s in sentences) {
      final joined = cur.isEmpty ? s : '$cur $s';
      if (cur.isNotEmpty && estimateSeconds(joined) > 20) {
        out.add(cur);
        cur = s;
      } else {
        cur = joined;
      }
    }
    if (cur.isNotEmpty) out.add(cur);
    return out;
  }

  /// Frame cap for a chunk: generous (2.5× the estimate) so slow speech
  /// isn't cut, but bounded so a runaway generation stops.
  static int _frameBudget(String text) =>
      (estimateSeconds(text) * 12.5 * 2.5 + 40).round().clamp(60, 512);
}
