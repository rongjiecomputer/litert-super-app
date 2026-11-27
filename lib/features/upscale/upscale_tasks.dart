import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;

import '../../core/inference/inference_worker.dart';
import 'ffmpeg.dart';
import 'tiler.dart';

/// RGBA pixels moved across isolates without copying.
class RgbaFrame {
  RgbaFrame(RgbImage image)
    : width = image.width,
      height = image.height,
      pixels = TransferableTypedData.fromList([image.toRgba()]);

  final int width, height;
  final TransferableTypedData pixels;

  /// Can only be called once (transfer semantics).
  Uint8List materialize() => pixels.materialize().asUint8List();
}

class TileProgress {
  const TileProgress(this.done, this.total);
  final int done, total;
}

class ImageUpscaleResult {
  ImageUpscaleResult({
    required this.inputSize,
    required this.outputSize,
    required this.inputPreview,
    required this.outputPreview,
    required this.tiles,
    required this.inferMs,
    required this.decodeMs,
    required this.preprocessMs,
    required this.postprocessMs,
    required this.totalMs,
  });

  final (int, int) inputSize, outputSize;
  final RgbaFrame inputPreview, outputPreview;
  final int tiles;
  final List<double> inferMs;
  final double decodeMs, preprocessMs, postprocessMs, totalMs;
}

const _lastImageKey = 'upscale.lastImage';

/// Largest preview side sent back to the UI (GPU texture limits).
const previewMaxSide = 8192;

class UpscaleImageTask extends WorkerTask<ImageUpscaleResult> {
  const UpscaleImageTask(this.path, this.overlap);

  final String path;
  final int overlap;

  @override
  Future<ImageUpscaleResult> run(WorkerContext ctx) async {
    final runner = ctx.requireRunner();
    final total = Stopwatch()..start();
    final src = decodeToRgb(File(path).readAsBytesSync());
    final decodeMs = total.elapsedMicroseconds / 1000;

    final tiler = TiledUpscaler.forSpec(ctx.spec!, overlap: overlap);
    var last = DateTime.now();
    final r = await tiler.upscale(
      src,
      runner.run,
      isCancelled: () => ctx.isCancelled,
      onTile: (done, all) {
        final now = DateTime.now();
        if (done == all || now.difference(last).inMilliseconds >= 80) {
          last = now;
          ctx.progress(TileProgress(done, all));
        }
      },
    );
    if (r.cancelled) throw const WorkerCancelled();
    ctx.scratch[_lastImageKey] = r.image;
    return ImageUpscaleResult(
      inputSize: (src.width, src.height),
      outputSize: (r.image.width, r.image.height),
      inputPreview: RgbaFrame(src.fitWithin(previewMaxSide)),
      outputPreview: RgbaFrame(r.image.fitWithin(previewMaxSide)),
      tiles: r.tiles,
      inferMs: r.inferMs,
      decodeMs: decodeMs,
      preprocessMs: r.preprocessMs,
      postprocessMs: r.postprocessMs,
      totalMs: total.elapsedMicroseconds / 1000,
    );
  }
}

/// Encodes the last upscaled image (kept in the worker) to PNG or JPEG.
class SaveLastImageTask extends WorkerTask<double> {
  const SaveLastImageTask(this.path);
  final String path;

  @override
  Future<double> run(WorkerContext ctx) async {
    final image = ctx.scratch[_lastImageKey] as RgbImage?;
    if (image == null) throw StateError('Nothing to save yet');
    final sw = Stopwatch()..start();
    final out = img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.data.buffer,
      numChannels: 3,
      order: img.ChannelOrder.rgb,
    );
    final ext = p.extension(path).toLowerCase();
    final bytes = (ext == '.jpg' || ext == '.jpeg')
        ? img.encodeJpg(out, quality: 95)
        : img.encodePng(out, level: 3);
    await File(path).writeAsBytes(bytes, flush: true);
    return sw.elapsedMicroseconds / 1000;
  }
}

RgbImage decodeToRgb(Uint8List bytes) {
  var decoded = img.decodeImage(bytes);
  if (decoded == null) throw const FormatException('Unsupported image');
  decoded = img.bakeOrientation(decoded);
  if (decoded.numChannels != 3 ||
      decoded.format != img.Format.uint8 ||
      decoded.hasPalette) {
    decoded = decoded.convert(format: img.Format.uint8, numChannels: 3);
  }
  final rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
  return RgbImage(
    decoded.width,
    decoded.height,
    Uint8List.fromList(rgb.sublist(0, decoded.width * decoded.height * 3)),
  );
}

// ---- Video -----------------------------------------------------------------

class VideoProgress {
  VideoProgress({
    required this.frame,
    required this.lastFrameMs,
    required this.tileDone,
    required this.tileTotal,
    required this.elapsedMs,
    this.inputPreview,
    this.outputPreview,
  });

  /// Frames fully encoded so far.
  final int frame;
  final double lastFrameMs;
  final int tileDone, tileTotal;
  final double elapsedMs;
  final RgbaFrame? inputPreview, outputPreview;
}

class VideoUpscaleResult {
  VideoUpscaleResult({
    required this.frames,
    required this.inputSize,
    required this.outputSize,
    required this.tiles,
    required this.inferMs,
    required this.preprocessMs,
    required this.postprocessMs,
    required this.encodeMs,
    required this.ioMs,
    required this.totalMs,
    required this.outputPath,
  });

  final int frames;
  final (int, int) inputSize, outputSize;
  final int tiles;
  final List<double> inferMs;
  final double preprocessMs, postprocessMs, encodeMs, ioMs, totalMs;
  final String outputPath;
}

/// Streams raw RGB frames out of ffmpeg, upscales each in place, and pipes
/// them into a second ffmpeg that encodes H.264 and muxes the source audio.
/// No intermediate files.
class UpscaleVideoTask extends WorkerTask<VideoUpscaleResult> {
  const UpscaleVideoTask(this.job);
  final VideoJob job;

  @override
  Future<VideoUpscaleResult> run(WorkerContext ctx) async {
    final runner = ctx.requireRunner();
    final tiler = TiledUpscaler.forSpec(ctx.spec!, overlap: job.overlap);
    final inW = job.decodeWidth, inH = job.decodeHeight;
    final outW = inW * tiler.scale, outH = inH * tiler.scale;
    final frameBytes = inW * inH * 3;

    final total = Stopwatch()..start();
    final dec = await Process.start(job.ffmpeg, job.decoderArgs());
    final enc = await Process.start(job.ffmpeg, job.encoderArgs(outW, outH));
    final decErr = _Tail(), encErr = _Tail();
    dec.stderr.listen(decErr.add);
    enc.stderr.listen(encErr.add);
    enc.stdout.drain<void>();

    final inferMs = <double>[];
    var pre = 0.0, post = 0.0, encodeMs = 0.0;
    var frames = 0, tiles = 0;
    var lastPreview = DateTime.fromMillisecondsSinceEpoch(0);
    var lastTileMsg = DateTime.fromMillisecondsSinceEpoch(0);
    var encoderBroken = false;

    Future<void> processFrame(Uint8List data) async {
      final frameSw = Stopwatch()..start();
      final src = RgbImage(inW, inH, data);
      final r = await tiler.upscale(
        src,
        runner.run,
        isCancelled: () => ctx.isCancelled,
        onTile: (done, all) {
          final now = DateTime.now();
          if (now.difference(lastTileMsg).inMilliseconds >= 120) {
            lastTileMsg = now;
            ctx.progress(
              VideoProgress(
                frame: frames,
                lastFrameMs: 0,
                tileDone: done,
                tileTotal: all,
                elapsedMs: total.elapsedMicroseconds / 1000,
              ),
            );
          }
        },
      );
      if (r.cancelled) return;
      inferMs.addAll(r.inferMs);
      pre += r.preprocessMs;
      post += r.postprocessMs;
      tiles += r.tiles;

      final encSw = Stopwatch()..start();
      try {
        enc.stdin.add(r.image.data);
        await enc.stdin.flush();
      } catch (_) {
        encoderBroken = true;
        return;
      }
      encodeMs += encSw.elapsedMicroseconds / 1000;
      frames++;

      final now = DateTime.now();
      final sendPreview = now.difference(lastPreview).inMilliseconds >= 1000;
      if (sendPreview) lastPreview = now;
      ctx.progress(
        VideoProgress(
          frame: frames,
          lastFrameMs: frameSw.elapsedMicroseconds / 1000,
          tileDone: r.tiles,
          tileTotal: r.tiles,
          elapsedMs: total.elapsedMicroseconds / 1000,
          inputPreview: sendPreview ? RgbaFrame(src) : null,
          outputPreview: sendPreview
              ? RgbaFrame(r.image.fitWithin(1920))
              : null,
        ),
      );
    }

    var frame = Uint8List(frameBytes);
    var filled = 0;
    try {
      await for (final chunk in dec.stdout) {
        var off = 0;
        while (off < chunk.length) {
          final n = math.min(frameBytes - filled, chunk.length - off);
          frame.setRange(filled, filled + n, chunk, off);
          filled += n;
          off += n;
          if (filled == frameBytes) {
            await processFrame(frame);
            frame = Uint8List(frameBytes);
            filled = 0;
            if (ctx.isCancelled || encoderBroken) break;
          }
        }
        if (ctx.isCancelled || encoderBroken) break;
      }
    } finally {
      if (ctx.isCancelled || encoderBroken) {
        dec.kill();
        enc.kill();
      }
    }

    if (ctx.isCancelled) {
      await enc.exitCode;
      _deleteQuietly(job.outputPath);
      throw const WorkerCancelled();
    }

    await enc.stdin.close().catchError((_) {});
    final decExit = await dec.exitCode;
    final encExit = await enc.exitCode;
    if (encoderBroken || encExit != 0) {
      _deleteQuietly(job.outputPath);
      throw ProcessException(
        job.ffmpeg,
        const ['(encoder)'],
        encErr.text.isEmpty ? 'Encoder failed' : encErr.text,
        encExit,
      );
    }
    if (decExit != 0 && frames == 0) {
      throw ProcessException(
        job.ffmpeg,
        const ['(decoder)'],
        decErr.text.isEmpty ? 'Decoder failed' : decErr.text,
        decExit,
      );
    }

    final totalMs = total.elapsedMicroseconds / 1000;
    final compute = inferMs.fold<double>(0, (a, b) => a + b) + pre + post;
    return VideoUpscaleResult(
      frames: frames,
      inputSize: (inW, inH),
      outputSize: (outW, outH),
      tiles: tiles,
      inferMs: inferMs,
      preprocessMs: pre,
      postprocessMs: post,
      encodeMs: encodeMs,
      ioMs: math.max(0, totalMs - compute - encodeMs),
      totalMs: totalMs,
      outputPath: job.outputPath,
    );
  }

  static void _deleteQuietly(String path) {
    try {
      final f = File(path);
      if (f.existsSync()) f.deleteSync();
    } catch (_) {}
  }
}

/// Keeps the last few KB of a process's stderr for error messages.
class _Tail {
  final _buf = <int>[];
  void add(List<int> data) {
    _buf.addAll(data);
    if (_buf.length > 4096) _buf.removeRange(0, _buf.length - 4096);
  }

  String get text => String.fromCharCodes(_buf).trim();
}
