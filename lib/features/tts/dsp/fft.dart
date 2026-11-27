import 'dart:math' as math;
import 'dart:typed_data';

/// In-place iterative radix-2 complex FFT (forward, e^{-i...} convention).
class Fft {
  Fft(this.n)
    : assert(n > 0 && (n & (n - 1)) == 0, 'n must be a power of two'),
      _cos = Float64List(n ~/ 2),
      _sin = Float64List(n ~/ 2),
      _rev = Int32List(n) {
    for (var i = 0; i < n ~/ 2; i++) {
      _cos[i] = math.cos(2 * math.pi * i / n);
      _sin[i] = -math.sin(2 * math.pi * i / n);
    }
    final bits = n.bitLength - 1;
    for (var i = 0; i < n; i++) {
      var r = 0;
      for (var b = 0; b < bits; b++) {
        if ((i >> b) & 1 == 1) r |= 1 << (bits - 1 - b);
      }
      _rev[i] = r;
    }
  }

  final int n;
  final Float64List _cos, _sin;
  final Int32List _rev;

  void transform(Float64List re, Float64List im) {
    for (var i = 0; i < n; i++) {
      final j = _rev[i];
      if (j > i) {
        var t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    for (var size = 2; size <= n; size <<= 1) {
      final half = size >> 1;
      final step = n ~/ size;
      for (var start = 0; start < n; start += size) {
        for (var k = 0; k < half; k++) {
          final wr = _cos[k * step], wi = _sin[k * step];
          final a = start + k, b = a + half;
          final xr = re[b] * wr - im[b] * wi;
          final xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

/// Periodic Hann window (`torch.hann_window(n)` default).
Float64List hannPeriodic(int n) => Float64List.fromList([
  for (var i = 0; i < n; i++) 0.5 - 0.5 * math.cos(2 * math.pi * i / n),
]);

/// Reflect-pads [x] by [left]/[right] samples (numpy/torch `reflect`).
Float32List reflectPad(Float32List x, int left, int right) {
  final n = x.length;
  final out = Float32List(n + left + right);
  int src(int i) {
    // Reflect without repeating the edge sample; fold repeatedly for
    // very short inputs.
    if (n == 1) return 0;
    final period = 2 * (n - 1);
    var j = i % period;
    if (j < 0) j += period;
    return j < n ? j : period - j;
  }

  for (var i = 0; i < out.length; i++) {
    out[i] = x[src(i - left)];
  }
  return out;
}
