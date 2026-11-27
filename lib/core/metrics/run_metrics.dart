import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

/// Summary statistics over a list of latencies (milliseconds).
class LatencyStats {
  const LatencyStats({
    required this.count,
    required this.mean,
    required this.p50,
    required this.p90,
    required this.min,
    required this.max,
    required this.sum,
  });

  final int count;
  final double mean, p50, p90, min, max, sum;

  static const empty = LatencyStats(
    count: 0,
    mean: 0,
    p50: 0,
    p90: 0,
    min: 0,
    max: 0,
    sum: 0,
  );

  factory LatencyStats.of(List<double> samples) {
    if (samples.isEmpty) return empty;
    final s = [...samples]..sort();
    double pct(double q) => s[math.min(s.length - 1, (q * s.length).floor())];
    final sum = s.fold<double>(0, (a, b) => a + b);
    return LatencyStats(
      count: s.length,
      mean: sum / s.length,
      p50: pct(0.5),
      p90: pct(0.9),
      min: s.first,
      max: s.last,
      sum: sum,
    );
  }
}

/// Facts about the currently loaded model (one per load).
class LoadInfo {
  const LoadInfo({
    required this.optionsSummary,
    required this.runtimeLabel,
    required this.requested,
    required this.actual,
    required this.didFallback,
    required this.fullyAccelerated,
    required this.loadMs,
    required this.warmupMs,
    required this.rssDeltaBytes,
    this.fallbackReason,
  });

  final String optionsSummary;
  final String runtimeLabel;
  final List<String> requested;
  final List<String> actual;
  final bool didFallback;
  final bool? fullyAccelerated;
  final double loadMs;
  final double warmupMs;
  final int rssDeltaBytes;
  final String? fallbackReason;
}

/// Cost of a single task run (image or video).
class RunMetrics {
  RunMetrics({
    required this.task,
    required this.at,
    required this.optionsSummary,
    required this.actualAccelerators,
    required this.inputSize,
    required this.outputSize,
    required this.tiles,
    required this.tileStats,
    required this.decodeMs,
    required this.preprocessMs,
    required this.postprocessMs,
    required this.totalMs,
    required this.rssBefore,
    required this.rssPeak,
    required this.rssAfter,
    required this.maxRss,
    this.loadMs,
    this.frames,
    this.cancelled = false,
  });

  final String task;
  final DateTime at;
  final String optionsSummary;
  final String actualAccelerators;
  final (int w, int h) inputSize;
  final (int w, int h) outputSize;
  final int tiles;
  final LatencyStats tileStats;
  final double decodeMs;
  final double preprocessMs;
  final double postprocessMs;
  final double totalMs;
  final int rssBefore, rssPeak, rssAfter, maxRss;

  /// Set when this run had to (re)load the model first.
  final double? loadMs;

  /// Video only.
  final int? frames;
  final bool cancelled;

  double get megapixelsOut => outputSize.$1 * outputSize.$2 / 1e6;
  double get tilesPerSecond =>
      tileStats.sum == 0 ? 0 : tiles / (tileStats.sum / 1000);
  double? get fps =>
      frames == null || totalMs == 0 ? null : frames! / (totalMs / 1000);
  int get rssDelta => rssPeak - rssBefore;
}

/// Samples process RSS on a timer to capture the peak during a task.
///
/// The worker isolate shares the process, so main-isolate sampling sees the
/// model's native allocations (weights, GPU staging, tensors) too.
class RssSampler {
  RssSampler({this.interval = const Duration(milliseconds: 50)});

  final Duration interval;
  Timer? _timer;
  int before = 0;
  int peak = 0;

  static int get current => ProcessInfo.currentRss;
  static int get max => ProcessInfo.maxRss;

  void start() {
    before = current;
    peak = before;
    _timer?.cancel();
    _timer = Timer.periodic(interval, (_) => sample());
  }

  void sample() {
    final now = current;
    if (now > peak) peak = now;
  }

  /// Stops sampling and returns `(before, peak, after)`.
  (int, int, int) stop() {
    _timer?.cancel();
    _timer = null;
    sample();
    return (before, peak, current);
  }
}

String formatBytes(num bytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  var v = bytes.toDouble();
  var sign = '';
  if (v < 0) {
    sign = '-';
    v = -v;
  }
  var i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return '$sign${v.toStringAsFixed(i == 0 ? 0 : 1)} ${units[i]}';
}

String formatMs(double ms) {
  if (ms >= 60000) {
    final m = ms ~/ 60000;
    final s = (ms % 60000) / 1000;
    return '${m}m ${s.toStringAsFixed(0)}s';
  }
  if (ms >= 1000) return '${(ms / 1000).toStringAsFixed(2)} s';
  if (ms >= 100) return '${ms.toStringAsFixed(0)} ms';
  return '${ms.toStringAsFixed(1)} ms';
}
