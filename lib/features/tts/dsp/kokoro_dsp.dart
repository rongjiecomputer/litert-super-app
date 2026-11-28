import 'dart:math' as math;
import 'dart:typed_data';

import 'fft.dart';

/// Host-side signal processing for Kokoro (StyleTTS2 + iSTFTNet): the final
/// iSTFT overlap-add that turns the vocoder graph's magnitude/phase
/// spectrograms into audio. (The LiteRT-LM vocoder graph computes the
/// harmonic source itself.)
///
/// Kokoro-82M's config: 24 kHz, n_fft 20, hop 5, periodic Hann window.
abstract final class KokoroDsp {
  static const sampleRate = 24000;

  /// Audio samples per acoustic frame (the vocoder's upsample factor 10·6
  /// times the iSTFT hop 5, times 2 for the F0/energy stride).
  static const samplesPerFrame = 600;
  static const nFft = 20;
  static const hop = 5;
  static const bins = nFft ~/ 2 + 1;

  static final _window = hannPeriodic(nFft);
  static final _cos = Float64List.fromList([
    for (var k = 0; k < bins; k++)
      for (var n = 0; n < nFft; n++) math.cos(2 * math.pi * k * n / nFft),
  ]);
  static final _sin = Float64List.fromList([
    for (var k = 0; k < bins; k++)
      for (var n = 0; n < nFft; n++) math.sin(2 * math.pi * k * n / nFft),
  ]);

  /// `torch.stft(n_fft=20, hop=5, hann, center=True, reflect)` → magnitude
  /// and phase (`atan2`), laid out `[22][frames]` (used by tests; the vocoder
  /// graph produces its own spectrograms).
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
