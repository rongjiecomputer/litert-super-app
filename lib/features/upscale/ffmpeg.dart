import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;

/// Located ffmpeg + ffprobe executables.
class FfmpegTools {
  const FfmpegTools(this.ffmpeg, this.ffprobe, this.version);

  final String ffmpeg;
  final String ffprobe;
  final String version;

  static String get _exe => Platform.isWindows ? '.exe' : '';

  /// Finds ffmpeg from [override] (exe or folder), then PATH, then
  /// well-known install folders (GUI apps on macOS don't inherit the shell
  /// PATH, so Homebrew locations are checked explicitly).
  static Future<FfmpegTools?> locate({String? override}) async {
    final dirs = <String>[];
    if (override != null) {
      dirs.add(
        FileSystemEntity.isDirectorySync(override)
            ? override
            : p.dirname(override),
      );
    }
    final pathVar = Platform.environment['PATH'] ?? '';
    dirs.addAll(pathVar.split(Platform.isWindows ? ';' : ':'));
    final env = Platform.environment;
    if (Platform.isWindows) {
      final local = env['LOCALAPPDATA'];
      final home = env['USERPROFILE'];
      dirs.addAll([
        if (local != null) p.join(local, 'Microsoft', 'WinGet', 'Links'),
        if (home != null) p.join(home, 'scoop', 'shims'),
        r'C:\ProgramData\chocolatey\bin',
        r'C:\ffmpeg\bin',
      ]);
    } else {
      dirs.addAll(['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin']);
    }
    for (final dir in dirs.where((d) => d.trim().isNotEmpty)) {
      final ffmpeg = p.join(dir.trim(), 'ffmpeg$_exe');
      final ffprobe = p.join(dir.trim(), 'ffprobe$_exe');
      if (File(ffmpeg).existsSync() && File(ffprobe).existsSync()) {
        try {
          final r = await Process.run(ffmpeg, ['-hide_banner', '-version']);
          if (r.exitCode == 0) {
            final first = LineSplitter.split('${r.stdout}').firstOrNull ?? '';
            return FfmpegTools(ffmpeg, ffprobe, first);
          }
        } catch (_) {
          // Try the next candidate.
        }
      }
    }
    return null;
  }

  Future<VideoInfo> probe(String path) async {
    final r = await Process.run(ffprobe, [
      '-v',
      'error',
      '-show_streams',
      '-show_format',
      '-of',
      'json',
      path,
    ]);
    if (r.exitCode != 0) {
      throw ProcessException(ffprobe, [path], '${r.stderr}', r.exitCode);
    }
    return VideoInfo.fromProbeJson(
      jsonDecode('${r.stdout}') as Map<String, Object?>,
    );
  }

  /// Decodes any audio (or video's audio track) to mono float PCM at
  /// [sampleRate]. Used by the TTS voice-cloning reference input.
  Future<(Float32List, int)> decodeAudio(String path, {int sampleRate = 24000}) async {
    final r = await Process.run(
      ffmpeg,
      ['-v', 'error', '-i', path, '-vn', '-f', 'f32le', '-ac', '1', '-ar', '$sampleRate', '-'],
      stdoutEncoding: null,
    );
    if (r.exitCode != 0) {
      throw ProcessException(ffmpeg, [path], '${r.stderr}', r.exitCode);
    }
    final bytes = Uint8List.fromList(r.stdout as List<int>);
    return (Float32List.fromList(bytes.buffer.asFloat32List(0, bytes.length ~/ 4)), sampleRate);
  }

  /// Best available H.264 encoder (falls back to MPEG-4 Part 2).
  Future<List<String>> videoCodecArgs() async {
    final r = await Process.run(ffmpeg, ['-hide_banner', '-encoders']);
    final list = '${r.stdout}';
    bool has(String name) => RegExp('\\s$name\\s').hasMatch(list);
    if (has('libx264')) {
      return ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18'];
    }
    if (Platform.isMacOS && has('h264_videotoolbox')) {
      return ['-c:v', 'h264_videotoolbox', '-b:v', '20M'];
    }
    if (Platform.isWindows && has('h264_mf')) {
      return ['-c:v', 'h264_mf', '-b:v', '20M'];
    }
    return ['-c:v', 'mpeg4', '-q:v', '2'];
  }
}

class VideoInfo {
  const VideoInfo({
    required this.width,
    required this.height,
    required this.fpsExpr,
    required this.fps,
    required this.hasAudio,
    required this.codec,
    this.frameCount,
    this.durationSec,
  });

  /// Display size (rotation applied).
  final int width, height;

  /// Raw rate expression for ffmpeg, e.g. `30000/1001`.
  final String fpsExpr;
  final double fps;
  final int? frameCount;
  final double? durationSec;
  final bool hasAudio;
  final String codec;

  int? get estimatedFrames =>
      frameCount ?? (durationSec == null ? null : (durationSec! * fps).round());

  factory VideoInfo.fromProbeJson(Map<String, Object?> json) {
    final streams = (json['streams'] as List? ?? const [])
        .cast<Map<String, Object?>>();
    final v = streams.firstWhere(
      (s) => s['codec_type'] == 'video',
      orElse: () => throw const FormatException('No video stream found'),
    );
    final hasAudio = streams.any((s) => s['codec_type'] == 'audio');

    var rotation = 0;
    final tags = v['tags'] as Map<String, Object?>?;
    if (tags?['rotate'] != null) {
      rotation = int.tryParse('${tags!['rotate']}') ?? 0;
    }
    for (final sd in (v['side_data_list'] as List? ?? const [])) {
      final r = (sd as Map<String, Object?>)['rotation'];
      if (r is num) rotation = r.toInt();
    }
    var w = (v['width'] as num).toInt(), h = (v['height'] as num).toInt();
    if (rotation.abs() % 180 == 90) (w, h) = (h, w);

    String rateExpr = '${v['avg_frame_rate'] ?? '0/0'}';
    double rate(String e) {
      final parts = e.split('/');
      final n = double.tryParse(parts[0]) ?? 0;
      final d = parts.length > 1 ? double.tryParse(parts[1]) ?? 1 : 1;
      return d == 0 ? 0 : n / d;
    }

    if (rate(rateExpr) <= 0) rateExpr = '${v['r_frame_rate'] ?? '30/1'}';
    if (rate(rateExpr) <= 0) rateExpr = '30/1';

    final format = json['format'] as Map<String, Object?>?;
    final duration = double.tryParse(
      '${v['duration'] ?? format?['duration'] ?? ''}',
    );
    return VideoInfo(
      width: w,
      height: h,
      fpsExpr: rateExpr,
      fps: rate(rateExpr),
      frameCount: int.tryParse('${v['nb_frames'] ?? ''}'),
      durationSec: duration,
      hasAudio: hasAudio,
      codec: '${v['codec_name'] ?? '?'}',
    );
  }
}

/// Everything the worker needs to run the decode → upscale → encode loop.
class VideoJob {
  const VideoJob({
    required this.ffmpeg,
    required this.inputPath,
    required this.outputPath,
    required this.decodeWidth,
    required this.decodeHeight,
    required this.fpsExpr,
    required this.maxFrames,
    required this.hasAudio,
    required this.codecArgs,
    required this.overlap,
  });

  final String ffmpeg;
  final String inputPath;
  final String outputPath;
  final int decodeWidth, decodeHeight;
  final String fpsExpr;

  /// 0 = all frames.
  final int maxFrames;
  final bool hasAudio;
  final List<String> codecArgs;
  final int overlap;

  List<String> decoderArgs() => [
    '-v', 'error', '-nostdin', //
    '-i', inputPath,
    '-an', '-sn',
    '-vf', 'scale=$decodeWidth:$decodeHeight:flags=area',
    if (maxFrames > 0) ...['-frames:v', '$maxFrames'],
    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-',
  ];

  List<String> encoderArgs(int outW, int outH) => [
    '-v', 'error', '-y', //
    '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', '${outW}x$outH', '-framerate', fpsExpr,
    '-i', '-',
    if (hasAudio) ...['-i', inputPath],
    '-map', '0:v:0',
    if (hasAudio) ...['-map', '1:a:0?', '-c:a', 'aac', '-b:a', '192k'],
    ...codecArgs,
    '-pix_fmt', 'yuv420p',
    '-shortest', '-movflags', '+faststart',
    outputPath,
  ];

  /// Even decode size for [info] scaled by [factor] (yuv420p friendly).
  static (int, int) decodeSize(VideoInfo info, double factor) {
    int even(double v) => ((v / 2).round() * 2).clamp(2, 1 << 15);
    return (even(info.width * factor), even(info.height * factor));
  }
}
