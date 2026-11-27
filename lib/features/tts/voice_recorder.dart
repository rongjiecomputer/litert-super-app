import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:record/record.dart';

/// Microphone capture for voice-cloning references (record package), as a
/// 24 kHz mono WAV in [tempDir].
class VoiceRecorder extends ChangeNotifier {
  VoiceRecorder(this.tempDir);

  final String tempDir;
  final _rec = AudioRecorder();
  StreamSubscription<Amplitude>? _amp;
  Timer? _timer;
  final _clock = Stopwatch();

  bool recording = false;
  Duration elapsed = Duration.zero;

  /// Input level 0..1 (from dBFS).
  double level = 0;
  String? error;

  Future<bool> start() async {
    error = null;
    try {
      if (!await _rec.hasPermission()) {
        error = 'Microphone permission denied';
        notifyListeners();
        return false;
      }
      Directory(tempDir).createSync(recursive: true);
      final path = p.join(
        tempDir,
        'reference-${DateTime.now().millisecondsSinceEpoch}.wav',
      );
      await _rec.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 24000,
          numChannels: 1,
          noiseSuppress: true,
        ),
        path: path,
      );
      recording = true;
      _clock
        ..reset()
        ..start();
      _timer = Timer.periodic(const Duration(milliseconds: 200), (_) {
        elapsed = _clock.elapsed;
        notifyListeners();
      });
      _amp = _rec
          .onAmplitudeChanged(const Duration(milliseconds: 100))
          .listen((a) {
            level = math.pow(10, a.current.clamp(-60, 0) / 40).toDouble();
            notifyListeners();
          });
      notifyListeners();
      return true;
    } catch (e) {
      error = 'Could not start recording: $e';
      recording = false;
      notifyListeners();
      return false;
    }
  }

  /// Stops and returns the WAV path (null if nothing was recorded).
  Future<String?> stop() async {
    _timer?.cancel();
    await _amp?.cancel();
    _amp = null;
    _clock.stop();
    String? path;
    try {
      path = await _rec.stop();
    } catch (e) {
      error = 'Recording failed: $e';
    }
    recording = false;
    level = 0;
    notifyListeners();
    return path;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _amp?.cancel();
    _rec.dispose();
    super.dispose();
  }
}
