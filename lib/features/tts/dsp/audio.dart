import 'dart:math' as math;
import 'dart:typed_data';

/// Mono float PCM plus its sample rate.
class PcmAudio {
  const PcmAudio(this.samples, this.sampleRate);
  final Float32List samples;
  final int sampleRate;

  double get seconds => samples.length / sampleRate;
}

/// Decodes a RIFF/WAVE file (PCM 8/16/24/32-bit int or 32/64-bit float,
/// incl. WAVE_FORMAT_EXTENSIBLE), down-mixing to mono.
PcmAudio decodeWav(Uint8List bytes) {
  final bd = ByteData.sublistView(bytes);
  String tag(int o) => String.fromCharCodes(bytes.sublist(o, o + 4));
  if (bytes.length < 12 || tag(0) != 'RIFF' || tag(8) != 'WAVE') {
    throw const FormatException('Not a WAV file');
  }
  int? format, channels, rate, bits;
  int? dataOff, dataLen;
  var p = 12;
  while (p + 8 <= bytes.length) {
    final id = tag(p);
    var len = bd.getUint32(p + 4, Endian.little);
    final body = p + 8;
    if (id == 'fmt ') {
      format = bd.getUint16(body, Endian.little);
      channels = bd.getUint16(body + 2, Endian.little);
      rate = bd.getUint32(body + 4, Endian.little);
      bits = bd.getUint16(body + 14, Endian.little);
      if (format == 0xfffe && len >= 26) {
        format = bd.getUint16(body + 24, Endian.little);
      }
    } else if (id == 'data') {
      // Streaming writers may leave the size unset.
      if (len == 0 || body + len > bytes.length) len = bytes.length - body;
      dataOff = body;
      dataLen = len;
      break;
    }
    p = body + len + (len.isOdd ? 1 : 0);
  }
  if (format == null || dataOff == null) {
    throw const FormatException('WAV is missing fmt/data chunks');
  }
  final ch = channels!, bps = bits! ~/ 8;
  final frames = dataLen! ~/ (bps * ch);
  final out = Float32List(frames);
  for (var i = 0; i < frames; i++) {
    var acc = 0.0;
    for (var c = 0; c < ch; c++) {
      final o = dataOff + (i * ch + c) * bps;
      acc += switch ((format, bits)) {
        (3, 32) => bd.getFloat32(o, Endian.little),
        (3, 64) => bd.getFloat64(o, Endian.little),
        (1, 8) => (bytes[o] - 128) / 128.0,
        (1, 16) => bd.getInt16(o, Endian.little) / 32768.0,
        (1, 24) =>
          ((bytes[o] | (bytes[o + 1] << 8) | (bytes[o + 2] << 16)) << 8)
                  .toSigned(32) /
              2147483648.0,
        (1, 32) => bd.getInt32(o, Endian.little) / 2147483648.0,
        _ => throw FormatException('Unsupported WAV format $format/$bits-bit'),
      };
    }
    out[i] = acc / ch;
  }
  return PcmAudio(out, rate!);
}

/// Encodes mono float PCM as 16-bit WAV.
Uint8List encodeWav16(Float32List samples, int sampleRate) {
  final dataLen = samples.length * 2;
  final b = ByteData(44 + dataLen);
  void str(int o, String s) {
    for (var i = 0; i < 4; i++) {
      b.setUint8(o + i, s.codeUnitAt(i));
    }
  }

  str(0, 'RIFF');
  b.setUint32(4, 36 + dataLen, Endian.little);
  str(8, 'WAVE');
  str(12, 'fmt ');
  b.setUint32(16, 16, Endian.little);
  b.setUint16(20, 1, Endian.little);
  b.setUint16(22, 1, Endian.little);
  b.setUint32(24, sampleRate, Endian.little);
  b.setUint32(28, sampleRate * 2, Endian.little);
  b.setUint16(32, 2, Endian.little);
  b.setUint16(34, 16, Endian.little);
  str(36, 'data');
  b.setUint32(40, dataLen, Endian.little);
  for (var i = 0; i < samples.length; i++) {
    final v = (samples[i].clamp(-1.0, 1.0) * 32767).round();
    b.setInt16(44 + i * 2, v, Endian.little);
  }
  return b.buffer.asUint8List();
}

/// Band-limited resampling with a Kaiser-windowed sinc kernel.
Float32List resample(Float32List x, int fromRate, int toRate) {
  if (fromRate == toRate || x.isEmpty) return Float32List.fromList(x);
  final ratio = toRate / fromRate;
  final outLen = (x.length * ratio).floor();
  final out = Float32List(outLen);
  // Cutoff at the lower Nyquist, slightly below to leave a transition band.
  final cutoff = 0.95 * math.min(1.0, ratio);
  const halfTaps = 24;
  const beta = 8.0;
  final i0b = _besselI0(beta);
  for (var j = 0; j < outLen; j++) {
    final t = j / ratio; // position in input samples
    final center = t.floor();
    var acc = 0.0, wsum = 0.0;
    for (var k = center - halfTaps + 1; k <= center + halfTaps; k++) {
      if (k < 0 || k >= x.length) continue;
      final d = t - k;
      final arg = d / halfTaps;
      if (arg.abs() >= 1) continue;
      final win = _besselI0(beta * math.sqrt(1 - arg * arg)) / i0b;
      final xx = math.pi * d * cutoff;
      final sinc = xx == 0 ? 1.0 : math.sin(xx) / xx;
      final w = cutoff * sinc * win;
      acc += x[k] * w;
      wsum += w;
    }
    // Normalising by the tap sum keeps unity DC gain at the edges too.
    out[j] = wsum == 0 ? 0 : acc / wsum;
  }
  return out;
}

double _besselI0(double x) {
  var sum = 1.0, term = 1.0;
  final q = x * x / 4;
  for (var k = 1; k < 50; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return sum;
}

/// Trims leading/trailing silence below [thresholdDb] relative to peak.
Float32List trimSilence(Float32List x, {double thresholdDb = -40, int sampleRate = 24000}) {
  if (x.isEmpty) return x;
  var peak = 0.0;
  for (final v in x) {
    peak = math.max(peak, v.abs());
  }
  if (peak == 0) return x;
  final thr = peak * math.pow(10, thresholdDb / 20);
  final win = sampleRate ~/ 100; // 10 ms
  int firstLoud() {
    for (var i = 0; i < x.length; i += win) {
      final end = math.min(i + win, x.length);
      for (var j = i; j < end; j++) {
        if (x[j].abs() > thr) return i;
      }
    }
    return x.length;
  }

  int lastLoud() {
    for (var i = x.length; i > 0; i -= win) {
      final start = math.max(0, i - win);
      for (var j = start; j < i; j++) {
        if (x[j].abs() > thr) return i;
      }
    }
    return 0;
  }

  final s = firstLoud(), e = lastLoud();
  if (e <= s) return x;
  return Float32List.sublistView(x, s, e);
}

/// Scales [x] so its peak is [peak] (no-op for silence).
Float32List normalizePeak(Float32List x, {double peak = 0.95}) {
  var m = 0.0;
  for (final v in x) {
    m = math.max(m, v.abs());
  }
  if (m == 0) return x;
  final g = peak / m;
  return Float32List.fromList([for (final v in x) v * g]);
}
