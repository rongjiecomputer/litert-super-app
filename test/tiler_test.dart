import 'dart:math';
import 'dart:typed_data';

import 'package:ai_litert/core/models/model_catalog.dart';
import 'package:ai_litert/core/models/model_spec.dart';
import 'package:ai_litert/features/upscale/tiler.dart';
import 'package:flutter_test/flutter_test.dart';

/// Fake model: nearest-neighbour ×[scale] from an NHWC tile to an NCHW
/// (or NHWC) output, matching the Real-ESRGAN tensor contract.
TileInference fakeNearest(TiledUpscaler t) {
  final ow = t.tileW * t.scale, oh = t.tileH * t.scale;
  return (Float32List input) {
    final out = Float32List(ow * oh * 3);
    for (var y = 0; y < oh; y++) {
      for (var x = 0; x < ow; x++) {
        final s = ((y ~/ t.scale) * t.tileW + (x ~/ t.scale)) * 3;
        for (var c = 0; c < 3; c++) {
          final d = t.outputLayout == TensorLayout.nchw
              ? c * ow * oh + y * ow + x
              : (y * ow + x) * 3 + c;
          out[d] = input[s + c];
        }
      }
    }
    return out;
  };
}

RgbImage randomImage(int w, int h, int seed) {
  final r = Random(seed);
  return RgbImage(
    w,
    h,
    Uint8List.fromList(List.generate(w * h * 3, (_) => r.nextInt(256))),
  );
}

RgbImage nearestReference(RgbImage src, int s) {
  final out = Uint8List(src.width * s * src.height * s * 3);
  final ow = src.width * s;
  for (var y = 0; y < src.height * s; y++) {
    for (var x = 0; x < ow; x++) {
      final si = ((y ~/ s) * src.width + x ~/ s) * 3;
      final di = (y * ow + x) * 3;
      out.setRange(di, di + 3, src.data, si);
    }
  }
  return RgbImage(ow, src.height * s, out);
}

void main() {
  group('TiledUpscaler', () {
    test('forSpec reads Real-ESRGAN geometry', () {
      final t = TiledUpscaler.forSpec(ModelCatalog.realEsrganX4v3, overlap: 8);
      expect(t.tileW, 128);
      expect(t.tileH, 128);
      expect(t.scale, 4);
      expect(t.stepW, 112);
      expect(t.inputLayout, TensorLayout.nhwc);
      expect(t.outputLayout, TensorLayout.nchw);
      expect(t.tilesFor(640, 480), 6 * 5);
    });

    for (final (w, h, overlap) in [
      (37, 23, 8), // smaller than one tile
      (128, 128, 0), // exactly one tile, no overlap
      (300, 129, 8), // edge tiles of 1px
      (257, 190, 16),
    ]) {
      test('stitch is exact for ${w}x$h overlap $overlap', () async {
        final t = TiledUpscaler(
          tileH: 128,
          tileW: 128,
          scale: 4,
          overlap: overlap,
          inputLayout: TensorLayout.nhwc,
          outputLayout: TensorLayout.nchw,
        );
        final src = randomImage(w, h, w * h);
        var calls = 0;
        final infer = fakeNearest(t);
        final r = await t.upscale(src, (x) {
          calls++;
          return infer(x);
        }, yieldBetweenTiles: false);
        expect(r.cancelled, isFalse);
        expect(r.tiles, t.tilesFor(w, h));
        expect(calls, r.tiles);
        expect(r.inferMs, hasLength(r.tiles));
        expect(r.image.width, w * 4);
        expect(r.image.height, h * 4);
        expect(r.image.data, nearestReference(src, 4).data);
      });
    }

    test('NHWC output layout is supported', () async {
      final t = TiledUpscaler(
        tileH: 16,
        tileW: 16,
        scale: 2,
        overlap: 2,
        inputLayout: TensorLayout.nhwc,
        outputLayout: TensorLayout.nhwc,
      );
      final src = randomImage(40, 21, 7);
      final r = await t.upscale(src, fakeNearest(t), yieldBetweenTiles: false);
      expect(r.image.data, nearestReference(src, 2).data);
    });

    test('clamps out-of-range and NaN model output', () async {
      final t = TiledUpscaler(
        tileH: 8,
        tileW: 8,
        scale: 1,
        overlap: 0,
        inputLayout: TensorLayout.nhwc,
        outputLayout: TensorLayout.nchw,
      );
      final values = [-0.5, double.nan, 2.0];
      final r = await t.upscale(
        randomImage(8, 8, 1),
        (_) => Float32List.fromList(
          List.generate(8 * 8 * 3, (i) => values[i ~/ 64]),
        ),
        yieldBetweenTiles: false,
      );
      expect(r.image.data.sublist(0, 3), [0, 0, 255]);
    });

    test('cancellation stops between tiles', () async {
      final t = TiledUpscaler.forSpec(ModelCatalog.realEsrganX4v3, overlap: 0);
      var calls = 0;
      final infer = fakeNearest(t);
      final r = await t.upscale(
        randomImage(512, 128, 3),
        (x) {
          calls++;
          return infer(x);
        },
        isCancelled: () => calls >= 2,
        yieldBetweenTiles: false,
      );
      expect(r.cancelled, isTrue);
      expect(calls, 2);
    });
  });

  test('RgbImage.toRgba and fitWithin', () {
    final img = randomImage(10, 6, 5);
    final rgba = img.toRgba();
    expect(rgba.length, 10 * 6 * 4);
    expect(rgba.sublist(0, 4), [...img.data.sublist(0, 3), 255]);
    final small = img.fitWithin(4);
    expect(small.width, 3);
    expect(small.height, 2);
    expect(identical(img.fitWithin(10), img), isTrue);
  });
}
