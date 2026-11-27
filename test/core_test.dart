import 'package:ai_litert/core/inference/inference_options.dart';
import 'package:ai_litert/core/metrics/run_metrics.dart';
import 'package:ai_litert/core/paths.dart';
import 'package:ai_litert/features/upscale/ffmpeg.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('AppPaths.resolveRoot', () {
    test('Windows uses USERPROFILE', () {
      expect(
        AppPaths.resolveRoot(
          isMobile: false,
          isWindows: true,
          environment: {'USERPROFILE': r'C:\Users\me'},
        ),
        r'C:\Users\me\.litert-lm',
      );
    });

    test('macOS/Linux use HOME', () {
      expect(
        AppPaths.resolveRoot(
          isMobile: false,
          isWindows: false,
          environment: {'HOME': '/Users/me'},
        ),
        '/Users/me/.litert-lm',
      );
    });

    test('mobile uses the app support dir', () {
      expect(
        AppPaths.resolveRoot(
          isMobile: true,
          isWindows: false,
          environment: const {},
          mobileBase: '/data/user/0/app/files',
        ),
        endsWith('litert-lm'),
      );
    });
  });

  group('InferenceOptions', () {
    test('JSON round-trip', () {
      const o = InferenceOptions(
        accelerator: AcceleratorKind.cpu,
        precision: PrecisionKind.fp16,
        cpuFallback: false,
        cpuThreads: 6,
      );
      expect(InferenceOptions.fromJson(o.toJson()), o);
    });

    test('unknown JSON values fall back to defaults', () {
      // `runtime` / `bufferMode` are legacy keys from the Interpreter era.
      final o = InferenceOptions.fromJson({
        'runtime': 'interpreter',
        'accelerator': 'nope',
        'cpuThreads': 2,
      });
      expect(o.accelerator, const InferenceOptions().accelerator);
      expect(o.cpuThreads, 2);
    });

    test('sanitized drops accelerators the platform lacks', () {
      final win = PlatformCaps.forOs('windows', 8);
      final o = const InferenceOptions(accelerator: AcceleratorKind.npu)
          .sanitized(win);
      expect(o.accelerator, AcceleratorKind.cpu);

      final threads = const InferenceOptions(cpuThreads: 64).sanitized(win);
      expect(threads.accelerator, AcceleratorKind.gpu);
      expect(threads.cpuThreads, 8);

      final mac = PlatformCaps.forOs('macos', 10);
      expect(
        const InferenceOptions(accelerator: AcceleratorKind.npu)
            .sanitized(mac)
            .accelerator,
        AcceleratorKind.npu,
      );
    });

    test('fp16 is reset where unsupported', () {
      final caps = PlatformCaps.forOs('windows', 4);
      final o = const InferenceOptions(
        accelerator: AcceleratorKind.cpu,
        precision: PrecisionKind.fp16,
      ).sanitized(caps);
      expect(o.precision, PrecisionKind.fp32);
    });
  });

  test('LatencyStats', () {
    final s = LatencyStats.of([5, 1, 3, 2, 4, 6, 7, 8, 9, 10]);
    expect(s.count, 10);
    expect(s.mean, 5.5);
    expect(s.min, 1);
    expect(s.max, 10);
    expect(s.p50, 6);
    expect(s.p90, 10);
    expect(LatencyStats.of(const []).count, 0);
  });

  group('VideoInfo.fromProbeJson', () {
    test('parses rate, rotation and audio', () {
      final info = VideoInfo.fromProbeJson({
        'streams': [
          {
            'codec_type': 'video',
            'codec_name': 'h264',
            'width': 1920,
            'height': 1080,
            'avg_frame_rate': '30000/1001',
            'nb_frames': '300',
            'side_data_list': [
              {'rotation': -90},
            ],
          },
          {'codec_type': 'audio'},
        ],
        'format': {'duration': '10.01'},
      });
      expect(info.width, 1080);
      expect(info.height, 1920);
      expect(info.fps, closeTo(29.97, 0.01));
      expect(info.fpsExpr, '30000/1001');
      expect(info.estimatedFrames, 300);
      expect(info.hasAudio, isTrue);
    });

    test('falls back to r_frame_rate and duration', () {
      final info = VideoInfo.fromProbeJson({
        'streams': [
          {
            'codec_type': 'video',
            'width': 640,
            'height': 360,
            'avg_frame_rate': '0/0',
            'r_frame_rate': '25/1',
          },
        ],
        'format': {'duration': '2.0'},
      });
      expect(info.fps, 25);
      expect(info.estimatedFrames, 50);
      expect(info.hasAudio, isFalse);
    });

    test('decodeSize is even', () {
      final info = VideoInfo.fromProbeJson({
        'streams': [
          {
            'codec_type': 'video',
            'width': 1279,
            'height': 719,
            'avg_frame_rate': '24/1',
          },
        ],
      });
      final (w, h) = VideoJob.decodeSize(info, 0.5);
      expect(w.isEven && h.isEven, isTrue);
      expect((w, h), (640, 360));
    });
  });
}
