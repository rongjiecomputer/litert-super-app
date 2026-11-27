import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import '../../core/inference/inference_options.dart';
import '../../core/inference/inference_worker.dart';
import '../../core/metrics/run_metrics.dart';
import '../../core/models/model_catalog.dart';
import '../../core/models/model_manager.dart';
import '../../core/settings.dart';
import 'ffmpeg.dart';
import 'tiler.dart';
import 'upscale_tasks.dart';

enum UpscaleBusy { idle, loading, image, video, saving }

class UpscaleController extends ChangeNotifier {
  UpscaleController({
    required this.settings,
    required this.models,
    required this.caps,
  }) {
    options = settings.inferenceOptions(useCaseId, caps);
    final params = settings.useCaseParams(useCaseId);
    overlap = (params['overlap'] as num?)?.toInt() ?? 8;
    videoScale = (params['videoScale'] as num?)?.toDouble() ?? 0.5;
    maxFrames = (params['maxFrames'] as num?)?.toInt() ?? 0;
  }

  static const useCaseId = 'upscale';
  final spec = ModelCatalog.realEsrganX4v3;
  final AppSettings settings;
  final ModelManager models;
  final PlatformCaps caps;

  // ---- configuration -------------------------------------------------------
  late InferenceOptions options;
  late int overlap;

  /// Video: decode scale applied before 4× upscaling (1.0 = native).
  late double videoScale;

  /// Video: 0 = all frames.
  late int maxFrames;

  InferenceOptions? loadedOptions;
  LoadInfo? loadInfo;
  bool get modelReady => models.isReady(spec);
  bool get needsReload => loadedOptions != options;

  // ---- job state -----------------------------------------------------------
  UpscaleBusy busy = UpscaleBusy.idle;
  String? status;
  double? progress;
  String? error;
  bool get isBusy => busy != UpscaleBusy.idle;

  // ---- image ---------------------------------------------------------------
  String? imagePath;
  ui.Image? inputImage, outputImage;
  (int, int)? imageInputSize, imageOutputSize;

  // ---- video ---------------------------------------------------------------
  FfmpegTools? ffmpeg;
  bool ffmpegChecked = false;
  String? videoPath;
  VideoInfo? videoInfo;
  VideoProgress? videoProgress;
  ui.Image? videoInputFrame, videoOutputFrame;
  String? videoOutputPath;

  // ---- metrics -------------------------------------------------------------
  RunMetrics? lastRun;
  final List<RunMetrics> history = [];

  InferenceWorker? _worker;
  final _rss = RssSampler();
  bool _disposed = false;

  TiledUpscaler get tiler => TiledUpscaler.forSpec(spec, overlap: overlap);

  void setOptions(InferenceOptions o) {
    options = o.sanitized(caps);
    settings.setInferenceOptions(useCaseId, options);
    _notify();
  }

  void setParams({int? overlap, double? videoScale, int? maxFrames}) {
    this.overlap = overlap ?? this.overlap;
    this.videoScale = videoScale ?? this.videoScale;
    this.maxFrames = maxFrames ?? this.maxFrames;
    settings.setUseCaseParams(useCaseId, {
      'overlap': this.overlap,
      'videoScale': this.videoScale,
      'maxFrames': this.maxFrames,
    });
    _notify();
  }

  // ---- model lifecycle -------------------------------------------------------

  Future<InferenceWorker> _ensureWorker() async =>
      _worker ??= await InferenceWorker.spawn();

  /// Loads the model with the current options if not loaded already.
  /// Returns the load time when a (re)load happened.
  Future<double?> _ensureLoaded() async {
    if (!modelReady) throw StateError('Download the model first (Models page)');
    if (!needsReload && loadInfo != null) return null;
    final w = await _ensureWorker();
    final requested = options;
    status = 'Loading model (${requested.summary})…';
    _notify();
    try {
      loadInfo = await w.run(
        LoadModelTask(models.pathOf(spec), spec, requested),
      );
      loadedOptions = requested;
    } catch (_) {
      loadInfo = null;
      loadedOptions = null;
      rethrow;
    }
    return loadInfo!.loadMs;
  }

  Future<void> loadModel() => _guard(UpscaleBusy.loading, () async {
    loadedOptions = null; // force reload
    _rss.start();
    await _ensureLoaded();
    _rss.stop();
    status = 'Model loaded on ${loadInfo!.actual.join(' + ')}';
  });

  Future<void> unloadModel() async {
    if (isBusy || _worker == null) return;
    await _worker!.run(const UnloadModelTask());
    loadInfo = null;
    loadedOptions = null;
    status = 'Model unloaded';
    _notify();
  }

  void cancel() {
    _worker?.cancel();
    status = 'Cancelling…';
    _notify();
  }

  // ---- image -----------------------------------------------------------------

  void setImage(String path) {
    if (isBusy) return;
    imagePath = path;
    _replace(input: null, output: null);
    imageInputSize = imageOutputSize = null;
    error = null;
    status = null;
    _notify();
  }

  Future<void> upscaleImage() async {
    final path = imagePath;
    if (path == null) return;
    await _guard(UpscaleBusy.image, () async {
      _rss.start();
      final loadMs = await _ensureLoaded();
      status = 'Upscaling ${p.basename(path)}…';
      progress = 0;
      _notify();
      final r = await _worker!.run(
        UpscaleImageTask(path, overlap),
        onProgress: (m) {
          if (m is TileProgress) {
            progress = m.done / m.total;
            status = 'Tile ${m.done}/${m.total}';
            _notify();
          }
        },
      );
      final (before, peak, after) = _rss.stop();
      final input = await _toUiImage(r.inputPreview);
      final output = await _toUiImage(r.outputPreview);
      _replace(input: input, output: output);
      imageInputSize = r.inputSize;
      imageOutputSize = r.outputSize;
      _record(
        RunMetrics(
          task: 'Image ${_size(r.inputSize)} → ${_size(r.outputSize)}',
          at: DateTime.now(),
          optionsSummary: loadedOptions!.summary,
          actualAccelerators: loadInfo!.actual.join(' + '),
          inputSize: r.inputSize,
          outputSize: r.outputSize,
          tiles: r.tiles,
          tileStats: LatencyStats.of(r.inferMs),
          decodeMs: r.decodeMs,
          preprocessMs: r.preprocessMs,
          postprocessMs: r.postprocessMs,
          totalMs: r.totalMs + (loadMs ?? 0),
          rssBefore: before,
          rssPeak: peak,
          rssAfter: after,
          maxRss: RssSampler.max,
          loadMs: loadMs,
        ),
      );
      status = 'Done in ${formatMs(lastRun!.totalMs)}';
    });
  }

  Future<void> saveImage(String path) => _guard(UpscaleBusy.saving, () async {
    status = 'Encoding ${p.extension(path).toUpperCase()}…';
    _notify();
    final ms = await _worker!.run(SaveLastImageTask(path));
    status = 'Saved ${p.basename(path)} (${formatMs(ms)})';
  });

  // ---- video -----------------------------------------------------------------

  Future<void> detectFfmpeg() async {
    ffmpeg = await FfmpegTools.locate(override: settings.ffmpegPath);
    ffmpegChecked = true;
    _notify();
  }

  Future<void> setVideo(String path) async {
    if (isBusy) return;
    videoPath = path;
    videoInfo = null;
    videoOutputPath = null;
    videoProgress = null;
    error = null;
    _replaceVideo(null, null);
    _notify();
    if (ffmpeg == null) await detectFfmpeg();
    if (ffmpeg == null) return;
    try {
      videoInfo = await ffmpeg!.probe(path);
    } catch (e) {
      error = 'ffprobe failed: $e';
    }
    _notify();
  }

  (int, int)? get videoDecodeSize =>
      videoInfo == null ? null : VideoJob.decodeSize(videoInfo!, videoScale);

  int? get videoFramesToProcess {
    final est = videoInfo?.estimatedFrames;
    if (maxFrames <= 0) return est;
    return est == null || est > maxFrames ? maxFrames : est;
  }

  String defaultVideoOutput() {
    final src = videoPath!;
    return p.join(p.dirname(src), '${p.basenameWithoutExtension(src)}_x4.mp4');
  }

  Future<void> upscaleVideo(String outputPath) async {
    final info = videoInfo, tools = ffmpeg, src = videoPath;
    if (info == null || tools == null || src == null) return;
    await _guard(UpscaleBusy.video, () async {
      _rss.start();
      final loadMs = await _ensureLoaded();
      status = 'Preparing encoder…';
      videoOutputPath = null;
      _notify();
      final (dw, dh) = videoDecodeSize!;
      final job = VideoJob(
        ffmpeg: tools.ffmpeg,
        inputPath: src,
        outputPath: outputPath,
        decodeWidth: dw,
        decodeHeight: dh,
        fpsExpr: info.fpsExpr,
        maxFrames: maxFrames,
        hasAudio: info.hasAudio,
        codecArgs: await tools.videoCodecArgs(),
        overlap: overlap,
      );
      final totalFrames = videoFramesToProcess;
      progress = 0;
      final r = await _worker!.run(
        UpscaleVideoTask(job),
        onProgress: (m) async {
          if (m is! VideoProgress) return;
          videoProgress = m;
          final frac = m.tileTotal == 0 ? 0 : m.tileDone / m.tileTotal;
          if (totalFrames != null && totalFrames > 0) {
            final done = m.frame + (m.lastFrameMs == 0 ? frac : 0);
            progress = (done / totalFrames).clamp(0, 1).toDouble();
          } else {
            progress = null;
          }
          status =
              'Frame ${m.frame}${totalFrames == null ? '' : '/$totalFrames'}'
              '${m.lastFrameMs == 0 ? ' · tile ${m.tileDone}/${m.tileTotal}' : ''}';
          if (m.inputPreview != null && m.outputPreview != null) {
            final a = await _toUiImage(m.inputPreview!);
            final b = await _toUiImage(m.outputPreview!);
            _replaceVideo(a, b);
          }
          _notify();
        },
      );
      final (before, peak, after) = _rss.stop();
      videoOutputPath = r.outputPath;
      _record(
        RunMetrics(
          task:
              'Video ${r.frames}f ${_size(r.inputSize)} → ${_size(r.outputSize)}',
          at: DateTime.now(),
          optionsSummary: loadedOptions!.summary,
          actualAccelerators: loadInfo!.actual.join(' + '),
          inputSize: r.inputSize,
          outputSize: r.outputSize,
          tiles: r.tiles,
          tileStats: LatencyStats.of(r.inferMs),
          decodeMs: r.ioMs,
          preprocessMs: r.preprocessMs,
          postprocessMs: r.postprocessMs + r.encodeMs,
          totalMs: r.totalMs + (loadMs ?? 0),
          rssBefore: before,
          rssPeak: peak,
          rssAfter: after,
          maxRss: RssSampler.max,
          loadMs: loadMs,
          frames: r.frames,
        ),
      );
      status =
          'Done: ${r.frames} frames in ${formatMs(r.totalMs)} '
          '(${lastRun!.fps!.toStringAsFixed(2)} fps)';
    });
  }

  // ---- helpers ---------------------------------------------------------------

  Future<void> _guard(UpscaleBusy kind, Future<void> Function() body) async {
    if (isBusy) return;
    busy = kind;
    error = null;
    progress = null;
    _notify();
    try {
      await body();
    } on WorkerCancelled {
      status = 'Cancelled';
    } catch (e) {
      error = '$e';
      status = null;
    } finally {
      _rss.stop();
      busy = UpscaleBusy.idle;
      progress = null;
      _notify();
    }
  }

  void _record(RunMetrics m) {
    lastRun = m;
    history.insert(0, m);
    if (history.length > 50) history.removeLast();
  }

  static String _size((int, int) s) => '${s.$1}×${s.$2}';

  void _replace({ui.Image? input, ui.Image? output}) {
    inputImage?.dispose();
    outputImage?.dispose();
    inputImage = input;
    outputImage = output;
  }

  void _replaceVideo(ui.Image? input, ui.Image? output) {
    videoInputFrame?.dispose();
    videoOutputFrame?.dispose();
    videoInputFrame = input;
    videoOutputFrame = output;
  }

  static Future<ui.Image> _toUiImage(RgbaFrame f) async {
    final buffer = await ui.ImmutableBuffer.fromUint8List(f.materialize());
    final descriptor = ui.ImageDescriptor.raw(
      buffer,
      width: f.width,
      height: f.height,
      pixelFormat: ui.PixelFormat.rgba8888,
    );
    final codec = await descriptor.instantiateCodec();
    final frame = await codec.getNextFrame();
    codec.dispose();
    descriptor.dispose();
    buffer.dispose();
    return frame.image;
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    _replace();
    _replaceVideo(null, null);
    _worker?.dispose();
    super.dispose();
  }
}
