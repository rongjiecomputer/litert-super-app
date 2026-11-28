import 'dart:math' as math;
import 'dart:typed_data';

import 'fft.dart';

/// Pitch-preserving time stretch with WSOLA (waveform-similarity overlap-add).
///
/// [rate] > 1 speeds the audio up (shorter output), < 1 slows it down. The
/// LiteRT-LM Kokoro graphs have no speed input, so the speaking rate is
/// applied to the synthesized speech instead.
Float32List timeStretch(
  Float32List x,
  double rate, {
  int sampleRate = 24000,
}) {
  if ((rate - 1).abs() < 1e-3 || x.length < 64) return x;
  // 25 ms frames at 50% overlap (periodic Hann sums to one), searching
  // ±10 ms around the nominal input position for the best-matching splice.
  final frame = (sampleRate * 0.025).round() ~/ 2 * 2;
  final hop = frame ~/ 2;
  final tolerance = (sampleRate * 0.010).round();
  final window = hannPeriodic(frame);
  final outLen = (x.length / rate).round();
  final out = Float64List(outLen + frame);
  final norm = Float64List(outLen + frame);
  final maxStart = math.max(0, x.length - frame);

  var prev = 0;
  for (var k = 0; k * hop < outLen; k++) {
    final nominal = (k * hop * rate).round().clamp(0, maxStart);
    var pos = nominal;
    if (k > 0) {
      // The signal that would naturally follow the previous segment.
      final natural = prev + hop;
      var best = -double.infinity;
      final lo = math.max(0, nominal - tolerance);
      final hi = math.min(maxStart, nominal + tolerance);
      if (natural + hop <= x.length) {
        for (var c = lo; c <= hi; c += 2) {
          var s = 0.0;
          for (var i = 0; i < hop; i += 2) {
            s += x[c + i] * x[natural + i];
          }
          if (s > best) {
            best = s;
            pos = c;
          }
        }
      }
    }
    final o = k * hop;
    for (var i = 0; i < frame && pos + i < x.length; i++) {
      out[o + i] += x[pos + i] * window[i];
      norm[o + i] += window[i];
    }
    prev = pos;
  }
  final y = Float32List(outLen);
  for (var i = 0; i < outLen; i++) {
    y[i] = norm[i] > 1e-3 ? out[i] / norm[i] : 0;
  }
  return y;
}
