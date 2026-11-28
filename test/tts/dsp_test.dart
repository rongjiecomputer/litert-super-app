import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ai_litert/features/tts/dsp/audio.dart';
import 'package:ai_litert/features/tts/dsp/fft.dart';
import 'package:ai_litert/features/tts/dsp/kokoro_dsp.dart';
import 'package:ai_litert/features/tts/dsp/mel.dart';
import 'package:ai_litert/features/tts/dsp/time_stretch.dart';
import 'package:ai_litert/features/tts/engines/sampling.dart';
import 'package:ai_litert/features/tts/text/npy.dart';
import 'package:flutter_test/flutter_test.dart';

import 'tts_test_utils.dart';

void main() {
  group('FFT', () {
    test('matches a direct DFT', () {
      const n = 64;
      final rng = math.Random(3);
      final x = List.generate(n, (_) => rng.nextDouble() - 0.5);
      final re = Float64List.fromList(x), im = Float64List(n);
      Fft(n).transform(re, im);
      for (final k in [0, 1, 7, 31, 32, 50]) {
        var sr = 0.0, si = 0.0;
        for (var i = 0; i < n; i++) {
          sr += x[i] * math.cos(2 * math.pi * k * i / n);
          si -= x[i] * math.sin(2 * math.pi * k * i / n);
        }
        expect(re[k], closeTo(sr, 1e-9));
        expect(im[k], closeTo(si, 1e-9));
      }
    });

    test('reflect padding mirrors without repeating the edge', () {
      final x = Float32List.fromList([1, 2, 3, 4]);
      expect(reflectPad(x, 2, 3), [3, 2, 1, 2, 3, 4, 3, 2, 1]);
    });
  });

  test('mel spectrogram matches the librosa/torch reference', () {
    final g = jsonDecode(File(golden('qwen_mel.json')).readAsStringSync())
        as Map<String, dynamic>;
    const sr = 24000;
    final n = (g['signal'] as Map)['n'] as int;
    final y = Float32List(n);
    for (var i = 0; i < n; i++) {
      final t = i / sr;
      y[i] = 0.5 * math.sin(2 * math.pi * (200 + 1800 * t) * t) +
          0.1 * math.sin(2 * math.pi * 3000 * t);
    }
    final mel = MelSpectrogram();
    final rowSums = (g['filter_row_sums'] as List).cast<num>();
    for (var m = 0; m < 128; m++) {
      final s = mel.filters[m].fold(0.0, (a, b) => a + b);
      expect(s, closeTo(rowSums[m].toDouble(), 1e-5), reason: 'filter $m');
    }
    final out = mel.compute(y);
    expect(out.length ~/ 128, g['frames']);
    for (final s in (g['samples'] as List).cast<Map<String, dynamic>>()) {
      final v = out[(s['frame'] as int) * 128 + (s['mel'] as int)];
      expect(
        v,
        closeTo((s['value'] as num).toDouble(), 2e-3),
        reason: 'frame ${s['frame']} mel ${s['mel']}',
      );
    }
  });

  group('Kokoro DSP', () {
    test('STFT → iSTFT reconstructs the signal', () {
      final rng = math.Random(7);
      final x = Float32List.fromList(
        List.generate(3000, (_) => rng.nextDouble() * 2 - 1),
      );
      final spec = KokoroDsp.stft(x);
      final frames = KokoroDsp.stftFrames(x.length);
      expect(spec.length, 2 * KokoroDsp.bins * frames);
      final mag = Float32List.sublistView(spec, 0, KokoroDsp.bins * frames);
      final phase = Float32List.sublistView(spec, KokoroDsp.bins * frames);
      final y = KokoroDsp.istft(mag, phase, frames, x.length);
      expect(y.length, x.length);
      for (var i = 0; i < x.length; i++) {
        expect(y[i], closeTo(x[i], 1e-4), reason: 'sample $i');
      }
    });
  });

  group('time stretch', () {
    // Zero crossings per second of a mono signal.
    double crossingsPerSecond(Float32List y) {
      var n = 0;
      for (var i = 1; i < y.length; i++) {
        if ((y[i - 1] < 0) != (y[i] < 0)) n++;
      }
      return n / (y.length / 24000);
    }

    final tone = Float32List.fromList([
      for (var i = 0; i < 24000; i++) 0.5 * math.sin(2 * math.pi * 220 * i / 24000),
    ]);

    test('rate 1 returns the input', () {
      expect(identical(timeStretch(tone, 1.0), tone), isTrue);
    });

    test('changes the duration but keeps the pitch and level', () {
      for (final rate in [0.6, 0.8, 1.5, 2.0]) {
        final y = timeStretch(tone, rate);
        expect(y.length, closeTo(tone.length / rate, 2), reason: 'rate $rate');
        // 220 Hz sine → 440 zero crossings per second, whatever the rate.
        expect(crossingsPerSecond(y), closeTo(440, 12), reason: 'rate $rate');
        var peak = 0.0;
        for (var i = 1000; i < y.length - 1000; i++) {
          peak = math.max(peak, y[i].abs());
        }
        expect(peak, closeTo(0.5, 0.06), reason: 'rate $rate');
      }
    });
  });

  group('audio', () {
    test('WAV 16-bit round trip', () {
      final x = Float32List.fromList([
        for (var i = 0; i < 480; i++) 0.8 * math.sin(i / 7),
      ]);
      final a = decodeWav(encodeWav16(x, 24000));
      expect(a.sampleRate, 24000);
      expect(a.samples.length, x.length);
      for (var i = 0; i < x.length; i++) {
        expect(a.samples[i], closeTo(x[i], 1 / 16000));
      }
    });

    test('resampling keeps a tone and its level', () {
      const from = 48000, to = 24000;
      final x = Float32List.fromList([
        for (var i = 0; i < from ~/ 4; i++)
          0.5 * math.sin(2 * math.pi * 1000 * i / from),
      ]);
      final y = resample(x, from, to);
      expect(y.length, x.length ~/ 2);
      var err = 0.0;
      for (var i = 200; i < y.length - 200; i++) {
        final ideal = 0.5 * math.sin(2 * math.pi * 1000 * i / to);
        err = math.max(err, (y[i] - ideal).abs());
      }
      expect(err, lessThan(0.01));
    });

    test('trimSilence drops quiet edges', () {
      final x = Float32List(24000)..fillRange(6000, 18000, 0.5);
      final y = trimSilence(x);
      expect(y.length, inInclusiveRange(12000, 12480));
    });
  });

  group('npy', () {
    test('float32 encode → parse round trip', () {
      final data = Float32List.fromList(List.generate(12, (i) => i * 0.5 - 2));
      final a = NpyArray.parse(encodeNpyF32(data, [3, 4]));
      expect(a.shape, [3, 4]);
      expect(a.data, data);
      expect(a.row(2), data.sublist(8));
    });

    test('half floats decode', () {
      expect(halfToFloat(0x3c00), 1.0);
      expect(halfToFloat(0xc000), -2.0);
      expect(halfToFloat(0x3555), closeTo(0.33325, 1e-5));
      expect(halfToFloat(0x0001), closeTo(5.96e-8, 1e-9));
    });

    final npz = modelFile('qwen3-tts-0.6b', 'tables/text_projection_fp32.npz');
    test(
      'reads the Qwen3-TTS projection npz',
      () {
        final m = NpyArray.readNpz(npz!);
        expect(m['w1']!.shape, [2048, 2048]);
        expect(m['b2']!.shape, [1024]);
      },
      skip: npz == null ? 'Qwen3-TTS not downloaded' : false,
    );
  });

  group('sampling', () {
    final logits = Float32List.fromList([0.1, 3.0, -1.0, 2.9, 0.0]);
    test('greedy and top-1 pick the argmax', () {
      expect(pickToken(logits, math.Random(1), greedy: true), 1);
      expect(pickToken(logits, math.Random(1), greedy: false, topK: 1), 1);
    });

    test('top-k never samples outside the top k', () {
      final rng = math.Random(5);
      final seen = <int>{};
      for (var i = 0; i < 500; i++) {
        seen.add(pickToken(logits, rng, greedy: false, topK: 2, temperature: 1));
      }
      expect(seen, {1, 3});
    });

    test('offset/length select a sub-range', () {
      final l = Float32List.fromList([9, 9, 0, 5, 1]);
      expect(pickToken(l, math.Random(1), greedy: true, offset: 2, length: 3), 1);
    });

    test('repetition penalty follows HF semantics', () {
      final s = Float32List.fromList([2, -2, 1]);
      applyRepetitionPenalty(s, [0, 1], 2);
      expect(s, [1, -4, 1]);
    });
  });
}
