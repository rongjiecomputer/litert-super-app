import 'dart:math' as math;
import 'dart:typed_data';

import 'fft.dart';

/// Host-side signal processing for the bucketed Kokoro (StyleTTS2 +
/// iSTFTNet) LiteRT graphs: the hn-NSF harmonic source + its STFT (the
/// vocoder's `har` input) and the final iSTFT overlap-add.
///
/// Ported from `kokoro/istftnet.py` (SineGen, SourceModuleHnNSF,
/// TorchSTFT) with Kokoro-82M's config: 24 kHz, upsample scale 300
/// (10·6·5), 8 harmonics, voiced threshold 10 Hz, n_fft 20, hop 5.
class KokoroDsp {
  KokoroDsp({required this.sourceWeights, required this.sourceBias})
    : assert(sourceWeights.length == harmonics);

  /// Kokoro-82M's `decoder.generator.m_source.l_linear` parameters
  /// (Apache-2.0, hexgrad/Kokoro-82M). They live outside the LiteRT graphs
  /// because the harmonic source is computed on the host.
  factory KokoroDsp.kokoro82m() => KokoroDsp(
    sourceWeights: Float32List.fromList(const [
      -0.08154186606407166, -0.1851966679096222, -0.18263398110866547, //
      -0.17837205529212952, -0.09873894602060318, 0.0826403945684433,
      0.08743999153375626, -0.39068546891212463, -0.5477443337440491,
    ]),
    sourceBias: -0.02945026010274887,
  );

  static const sampleRate = 24000;
  static const upsample = 300;
  static const harmonics = 9; // fundamental + 8 overtones
  static const voicedThreshold = 10.0;
  static const sineAmp = 0.1;
  static const noiseStd = 0.003;
  static const nFft = 20;
  static const hop = 5;
  static const bins = nFft ~/ 2 + 1;

  /// `m_source.l_linear` (9 → 1) weights/bias.
  final Float32List sourceWeights;
  final double sourceBias;

  static final _window = hannPeriodic(nFft);
  static final _cos = Float64List.fromList([
    for (var k = 0; k < bins; k++)
      for (var n = 0; n < nFft; n++) math.cos(2 * math.pi * k * n / nFft),
  ]);
  static final _sin = Float64List.fromList([
    for (var k = 0; k < bins; k++)
      for (var n = 0; n < nFft; n++) math.sin(2 * math.pi * k * n / nFft),
  ]);

  /// Builds the vocoder's `har` input `[22][frames]` (magnitude bins then
  /// phase bins, each time-major) from frame-rate F0 (`[2·L]`).
  Float32List harmonicSource(Float32List f0, math.Random rng) {
    final src = _sourceSignal(f0, rng);
    return stft(src);
  }

  /// SourceModuleHnNSF: merged, tanh-squashed harmonic excitation at 24 kHz.
  Float32List _sourceSignal(Float32List f0, math.Random rng) {
    final frames = f0.length;
    final n = frames * upsample;
    final out = Float64List(n);
    final gauss = _Gaussian(rng);
    // Per harmonic: phase accumulates at frame rate, then is linearly
    // interpolated to sample rate (matches SineGen's interpolate trick; the
    // random initial phase it adds at t=0 falls between interpolation taps
    // and has no effect, so it is omitted).
    final phaseFrames = Float64List(frames);
    for (var h = 1; h <= harmonics; h++) {
      var acc = 0.0;
      for (var i = 0; i < frames; i++) {
        final r = f0[i] * h / sampleRate;
        acc += r - r.floorToDouble();
        phaseFrames[i] = acc * 2 * math.pi * upsample;
      }
      final w = sourceWeights[h - 1];
      for (var j = 0; j < n; j++) {
        var s = (j + 0.5) / upsample - 0.5;
        if (s < 0) s = 0;
        final i0 = s.floor();
        final lam = s - i0;
        final i1 = i0 + 1 < frames ? i0 + 1 : frames - 1;
        final phase = phaseFrames[i0] * (1 - lam) + phaseFrames[i1] * lam;
        final voiced = f0[j ~/ upsample] > voicedThreshold;
        final sine = voiced ? math.sin(phase) * sineAmp : 0.0;
        final noise =
            (voiced ? noiseStd : sineAmp / 3) * gauss.next();
        out[j] += w * (sine + noise);
      }
    }
    final merged = Float32List(n);
    for (var j = 0; j < n; j++) {
      merged[j] = _tanh(out[j] + sourceBias);
    }
    return merged;
  }

  /// `torch.stft(n_fft=20, hop=5, hann, center=True, reflect)` → magnitude
  /// and phase (`atan2`), laid out `[22][frames]`.
  static Float32List stft(Float32List x) {
    final padded = reflectPad(x, nFft ~/ 2, nFft ~/ 2);
    final frames = 1 + (padded.length - nFft) ~/ hop;
    final out = Float32List(2 * bins * frames);
    final seg = Float64List(nFft);
    for (var f = 0; f < frames; f++) {
      final off = f * hop;
      for (var i = 0; i < nFft; i++) {
        seg[i] = padded[off + i] * _window[i];
      }
      for (var k = 0; k < bins; k++) {
        var re = 0.0, im = 0.0;
        final base = k * nFft;
        for (var i = 0; i < nFft; i++) {
          re += seg[i] * _cos[base + i];
          im -= seg[i] * _sin[base + i];
        }
        out[k * frames + f] = math.sqrt(re * re + im * im);
        out[(bins + k) * frames + f] = math.atan2(im, re);
      }
    }
    return out;
  }

  /// Number of STFT frames for [samples] source samples.
  static int stftFrames(int samples) => samples ~/ hop + 1;

  /// `torch.istft(mag·e^{i·phase}, n_fft 20, hop 5, hann, center=True)` of
  /// the vocoder outputs (both `[11][frames]`), trimmed to [samples].
  ///
  /// (The `istft_W*.bin` bases shipped next to the legacy graphs do not
  /// reproduce the reference waveform; this direct form does — waveform
  /// correlation 0.997 vs the ONNX export.)
  static Float32List istft(
    Float32List mag,
    Float32List phase,
    int frames,
    int samples,
  ) {
    final raw = Float64List((frames - 1) * hop + nFft);
    final env = Float64List(raw.length);
    final w = _window;
    final seg = Float64List(nFft);
    for (var f = 0; f < frames; f++) {
      // Real inverse DFT of the one-sided spectrum (DC/Nyquist imaginary
      // parts are ignored, as irfft does).
      for (var i = 0; i < nFft; i++) {
        seg[i] = 0;
      }
      for (var k = 0; k < bins; k++) {
        final m = mag[k * frames + f];
        if (m == 0) continue;
        final ph = phase[k * frames + f];
        final re = m * math.cos(ph), im = m * math.sin(ph);
        final scale = (k == 0 || k == bins - 1) ? 1.0 : 2.0;
        final base = k * nFft;
        for (var i = 0; i < nFft; i++) {
          seg[i] += scale * (re * _cos[base + i] - im * _sin[base + i]);
        }
      }
      final off = f * hop;
      for (var i = 0; i < nFft; i++) {
        raw[off + i] += seg[i] / nFft * w[i];
        env[off + i] += w[i] * w[i];
      }
    }
    const start = nFft ~/ 2;
    final len = math.max(0, math.min(samples, raw.length - start));
    final out = Float32List(len);
    for (var i = 0; i < len; i++) {
      final e = env[start + i];
      out[i] = e > 1e-11 ? raw[start + i] / e : 0;
    }
    return out;
  }
}

double _tanh(double x) {
  if (x > 20) return 1;
  if (x < -20) return -1;
  final e = math.exp(2 * x);
  return (e - 1) / (e + 1);
}

/// Box–Muller standard normal samples.
class _Gaussian {
  _Gaussian(this._rng);
  final math.Random _rng;
  double? _spare;

  double next() {
    final s = _spare;
    if (s != null) {
      _spare = null;
      return s;
    }
    double u;
    do {
      u = _rng.nextDouble();
    } while (u <= 1e-12);
    final v = _rng.nextDouble();
    final r = math.sqrt(-2 * math.log(u));
    _spare = r * math.sin(2 * math.pi * v);
    return r * math.cos(2 * math.pi * v);
  }
}
