import 'dart:typed_data';

import '../../core/models/model_spec.dart';

/// Packed 8-bit RGB image, row-major, no padding.
class RgbImage {
  RgbImage(this.width, this.height, this.data)
    : assert(data.length == width * height * 3);

  final int width;
  final int height;
  final Uint8List data;

  Uint8List toRgba() {
    final out = Uint8List(width * height * 4);
    for (var i = 0, j = 0; i < data.length; i += 3, j += 4) {
      out[j] = data[i];
      out[j + 1] = data[i + 1];
      out[j + 2] = data[i + 2];
      out[j + 3] = 255;
    }
    return out;
  }

  /// Nearest-neighbour downscale so the longest side is <= [maxSide].
  RgbImage fitWithin(int maxSide) {
    final longest = width > height ? width : height;
    if (longest <= maxSide) return this;
    final factor = (longest + maxSide - 1) ~/ maxSide;
    final w = width ~/ factor, h = height ~/ factor;
    final out = Uint8List(w * h * 3);
    for (var y = 0; y < h; y++) {
      final srcRow = y * factor * width;
      for (var x = 0; x < w; x++) {
        final s = (srcRow + x * factor) * 3, d = (y * w + x) * 3;
        out[d] = data[s];
        out[d + 1] = data[s + 1];
        out[d + 2] = data[s + 2];
      }
    }
    return RgbImage(w, h, out);
  }
}

typedef TileInference = Float32List Function(Float32List input);

class TiledResult {
  TiledResult({
    required this.image,
    required this.tiles,
    required this.inferMs,
    required this.preprocessMs,
    required this.postprocessMs,
    required this.cancelled,
  });

  final RgbImage image;
  final int tiles;
  final List<double> inferMs;
  final double preprocessMs;
  final double postprocessMs;
  final bool cancelled;
}

/// Splits an image into fixed-size model tiles with [overlap] pixels of
/// context on every side, runs the model per tile, and stitches the
/// centre of every output tile back together.
///
/// Pixels outside the image are clamp-to-edge replicated, so edge tiles and
/// images smaller than one tile work without special cases.
class TiledUpscaler {
  TiledUpscaler({
    required this.tileH,
    required this.tileW,
    required this.scale,
    required this.overlap,
    required this.inputLayout,
    required this.outputLayout,
  }) : assert(tileW - 2 * overlap > 0 && tileH - 2 * overlap > 0);

  factory TiledUpscaler.forSpec(ModelSpec spec, {int overlap = 8}) {
    final (ih, iw, ic) = ModelSpec.hwc(spec.inputShape, spec.inputLayout);
    final (oh, ow, oc) = ModelSpec.hwc(spec.outputShape, spec.outputLayout);
    if (ic != 3 || oc != 3 || oh % ih != 0 || oh ~/ ih != ow ~/ iw) {
      throw ArgumentError('Unsupported upscaler spec ${spec.id}');
    }
    final maxOverlap = (ih < iw ? ih : iw) ~/ 2 - 1;
    return TiledUpscaler(
      tileH: ih,
      tileW: iw,
      scale: oh ~/ ih,
      overlap: overlap.clamp(0, maxOverlap),
      inputLayout: spec.inputLayout,
      outputLayout: spec.outputLayout,
    );
  }

  final int tileH, tileW, scale, overlap;
  final TensorLayout inputLayout, outputLayout;

  int get stepW => tileW - 2 * overlap;
  int get stepH => tileH - 2 * overlap;

  int tilesFor(int width, int height) =>
      ((width + stepW - 1) ~/ stepW) * ((height + stepH - 1) ~/ stepH);

  /// Upscales [src]. Yields to the event loop between tiles when
  /// [yieldBetweenTiles] is set, so the hosting isolate can process
  /// cancel messages.
  Future<TiledResult> upscale(
    RgbImage src,
    TileInference infer, {
    void Function(int done, int total)? onTile,
    bool Function()? isCancelled,
    bool yieldBetweenTiles = true,
  }) async {
    final outW = src.width * scale, outH = src.height * scale;
    final dst = Uint8List(outW * outH * 3);
    final input = Float32List(tileH * tileW * 3);
    final total = tilesFor(src.width, src.height);
    final inferMs = <double>[];
    final sw = Stopwatch();
    var pre = 0.0, post = 0.0;
    var done = 0;
    var cancelled = false;

    outer:
    for (var cy = 0; cy < src.height; cy += stepH) {
      for (var cx = 0; cx < src.width; cx += stepW) {
        if (isCancelled?.call() ?? false) {
          cancelled = true;
          break outer;
        }
        final cw = (src.width - cx) < stepW ? src.width - cx : stepW;
        final ch = (src.height - cy) < stepH ? src.height - cy : stepH;

        sw
          ..reset()
          ..start();
        _fillInput(src, cx - overlap, cy - overlap, input);
        pre += _ms(sw);

        sw
          ..reset()
          ..start();
        final output = infer(input);
        inferMs.add(_ms(sw));

        sw
          ..reset()
          ..start();
        _writeOutput(output, cx, cy, cw, ch, dst, outW);
        post += _ms(sw);

        done++;
        onTile?.call(done, total);
        if (yieldBetweenTiles) await Future<void>.delayed(Duration.zero);
      }
    }

    return TiledResult(
      image: RgbImage(outW, outH, dst),
      tiles: done,
      inferMs: inferMs,
      preprocessMs: pre,
      postprocessMs: post,
      cancelled: cancelled,
    );
  }

  static double _ms(Stopwatch sw) => sw.elapsedMicroseconds / 1000.0;

  void _fillInput(RgbImage src, int sx, int sy, Float32List input) {
    final w = src.width, h = src.height, data = src.data;
    const inv = 1.0 / 255.0;
    final plane = tileH * tileW;
    for (var yy = 0; yy < tileH; yy++) {
      var y = sy + yy;
      if (y < 0) y = 0;
      if (y >= h) y = h - 1;
      final row = y * w;
      for (var xx = 0; xx < tileW; xx++) {
        var x = sx + xx;
        if (x < 0) x = 0;
        if (x >= w) x = w - 1;
        final s = (row + x) * 3;
        if (inputLayout == TensorLayout.nhwc) {
          final d = (yy * tileW + xx) * 3;
          input[d] = data[s] * inv;
          input[d + 1] = data[s + 1] * inv;
          input[d + 2] = data[s + 2] * inv;
        } else {
          final d = yy * tileW + xx;
          input[d] = data[s] * inv;
          input[plane + d] = data[s + 1] * inv;
          input[2 * plane + d] = data[s + 2] * inv;
        }
      }
    }
  }

  void _writeOutput(
    Float32List out,
    int cx,
    int cy,
    int cw,
    int ch,
    Uint8List dst,
    int dstW,
  ) {
    final ow = tileW * scale, oh = tileH * scale;
    final plane = ow * oh;
    final off = overlap * scale;
    final rows = ch * scale, cols = cw * scale;
    for (var oy = 0; oy < rows; oy++) {
      final my = off + oy;
      final dRow = ((cy * scale + oy) * dstW + cx * scale) * 3;
      for (var ox = 0; ox < cols; ox++) {
        final mx = off + ox;
        final d = dRow + ox * 3;
        if (outputLayout == TensorLayout.nchw) {
          final i = my * ow + mx;
          dst[d] = _toByte(out[i]);
          dst[d + 1] = _toByte(out[plane + i]);
          dst[d + 2] = _toByte(out[2 * plane + i]);
        } else {
          final i = (my * ow + mx) * 3;
          dst[d] = _toByte(out[i]);
          dst[d + 1] = _toByte(out[i + 1]);
          dst[d + 2] = _toByte(out[i + 2]);
        }
      }
    }
  }

  /// Clamp 0..1 float to a byte; NaN maps to 0.
  static int _toByte(double v) {
    if (!(v > 0)) return 0;
    if (v >= 1) return 255;
    return (v * 255 + 0.5).toInt();
  }
}
