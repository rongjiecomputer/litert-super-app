import 'dart:math' as math;
import 'dart:typed_data';

import 'fft.dart';

/// Log-mel spectrogram matching Qwen3-TTS's speaker-encoder front end
/// (`modeling_qwen3_tts.mel_spectrogram`): 24 kHz, n_fft 1024, periodic
/// Hann 1024, hop 256, reflect pad (n_fft-hop)/2, center=False, magnitude
/// `sqrt(re²+im²+1e-9)`, Slaney mel 128 bins 0–12 kHz, `log(max(x,1e-5))`.
class MelSpectrogram {
  MelSpectrogram({
    this.sampleRate = 24000,
    this.nFft = 1024,
    this.hop = 256,
    this.nMels = 128,
    this.fMin = 0,
    this.fMax = 12000,
  }) : _fft = Fft(nFft),
       _window = hannPeriodic(nFft),
       filters = slaneyMelFilters(
         sampleRate: sampleRate,
         nFft: nFft,
         nMels: nMels,
         fMin: fMin,
         fMax: fMax,
       );

  final int sampleRate, nFft, hop, nMels;
  final double fMin, fMax;
  final Fft _fft;
  final Float64List _window;

  /// `[nMels][nFft/2+1]` filterbank.
  final List<Float64List> filters;

  int framesFor(int samples) {
    final pad = (nFft - hop) ~/ 2;
    return 1 + (samples + 2 * pad - nFft) ~/ hop;
  }

  /// Returns the log-mel spectrogram as `[frames][nMels]` row-major
  /// (time-major, the layout the speaker encoder consumes).
  Float32List compute(Float32List audio) {
    final pad = (nFft - hop) ~/ 2;
    final y = reflectPad(audio, pad, pad);
    final frames = 1 + (y.length - nFft) ~/ hop;
    final bins = nFft ~/ 2 + 1;
    final out = Float32List(frames * nMels);
    final re = Float64List(nFft), im = Float64List(nFft);
    final mag = Float64List(bins);
    for (var f = 0; f < frames; f++) {
      final off = f * hop;
      for (var i = 0; i < nFft; i++) {
        re[i] = y[off + i] * _window[i];
        im[i] = 0;
      }
      _fft.transform(re, im);
      for (var k = 0; k < bins; k++) {
        mag[k] = math.sqrt(re[k] * re[k] + im[k] * im[k] + 1e-9);
      }
      for (var m = 0; m < nMels; m++) {
        final w = filters[m];
        var s = 0.0;
        for (var k = 0; k < bins; k++) {
          s += w[k] * mag[k];
        }
        out[f * nMels + m] = math.log(math.max(s, 1e-5));
      }
    }
    return out;
  }
}

double _hzToMel(double f) {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  final logStep = math.log(6.4) / 27.0;
  return f >= minLogHz
      ? minLogMel + math.log(f / minLogHz) / logStep
      : f / fSp;
}

double _melToHz(double m) {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  final logStep = math.log(6.4) / 27.0;
  return m >= minLogMel
      ? minLogHz * math.exp(logStep * (m - minLogMel))
      : fSp * m;
}

/// `librosa.filters.mel(htk=False, norm='slaney')`.
List<Float64List> slaneyMelFilters({
  required int sampleRate,
  required int nFft,
  required int nMels,
  required double fMin,
  required double fMax,
}) {
  final bins = nFft ~/ 2 + 1;
  final fftFreqs = [for (var k = 0; k < bins; k++) k * sampleRate / nFft];
  final melMin = _hzToMel(fMin), melMax = _hzToMel(fMax);
  final melF = [
    for (var i = 0; i < nMels + 2; i++)
      _melToHz(melMin + (melMax - melMin) * i / (nMels + 1)),
  ];
  final out = <Float64List>[];
  for (var m = 0; m < nMels; m++) {
    final w = Float64List(bins);
    final d0 = melF[m + 1] - melF[m];
    final d1 = melF[m + 2] - melF[m + 1];
    final enorm = 2.0 / (melF[m + 2] - melF[m]);
    for (var k = 0; k < bins; k++) {
      final lower = (fftFreqs[k] - melF[m]) / d0;
      final upper = (melF[m + 2] - fftFreqs[k]) / d1;
      w[k] = math.max(0, math.min(lower, upper)) * enorm;
    }
    out.add(w);
  }
  return out;
}
