import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_soloud/flutter_soloud.dart';

/// Streaming PCM player for synthesized speech (flutter_soloud).
///
/// Chunks are appended to a SoLoud buffer stream as the engine produces
/// them; playback starts once [preBufferSeconds] are queued and SoLoud
/// pauses/resumes by itself on underrun (Qwen3 on CPU is often slower than
/// real time). The whole clip is kept for replay and saving.
class TtsPlayer extends ChangeNotifier {
  TtsPlayer({this.sampleRate = 24000, this.preBufferSeconds = 1.5});

  final int sampleRate;
  final double preBufferSeconds;

  AudioSource? _source;
  SoundHandle? _handle;
  final List<Float32List> _chunks = [];
  int _samples = 0;
  bool _streaming = false;
  bool _autoplay = true;
  Timer? _ticker;
  String? error;

  /// Playback position within the clip.
  Duration position = Duration.zero;

  bool buffering = false;

  bool get hasAudio => _samples > 0;
  bool get streaming => _streaming;
  double get seconds => _samples / sampleRate;
  bool get playing => _handle != null && !_paused;
  bool _paused = false;

  Future<bool> _ensureInit() async {
    final s = SoLoud.instance;
    if (s.isInitialized) return true;
    try {
      await s.init(sampleRate: 48000, channels: Channels.stereo);
      return true;
    } catch (e) {
      error = 'Audio output unavailable: $e';
      notifyListeners();
      return false;
    }
  }

  /// Starts a new clip (stops and drops the previous one).
  Future<void> begin({bool autoplay = true}) async {
    await stop();
    await _disposeSource();
    _chunks.clear();
    _samples = 0;
    position = Duration.zero;
    error = null;
    _autoplay = autoplay;
    if (!await _ensureInit()) return;
    _source = SoLoud.instance.setBufferStream(
      maxBufferSizeDuration: const Duration(minutes: 30),
      bufferingType: BufferingType.preserved,
      bufferingTimeNeeds: preBufferSeconds,
      sampleRate: sampleRate,
      channels: Channels.mono,
      format: BufferType.f32le,
      onBuffering: (isBuffering, handle, time) {
        buffering = isBuffering;
        notifyListeners();
      },
    );
    _streaming = true;
    notifyListeners();
  }

  /// Appends a chunk of mono float PCM.
  void add(Float32List pcm) {
    if (pcm.isEmpty) return;
    _chunks.add(pcm);
    _samples += pcm.length;
    final src = _source;
    if (src != null && _streaming) {
      SoLoud.instance.addAudioDataStream(
        src,
        pcm.buffer.asUint8List(pcm.offsetInBytes, pcm.lengthInBytes),
      );
      if (_autoplay && _handle == null && seconds >= preBufferSeconds) {
        _startPlayback();
      }
    }
    notifyListeners();
  }

  /// Marks the clip complete (also after cancel/error).
  void end() {
    final src = _source;
    if (src != null && _streaming) {
      SoLoud.instance.setDataIsEnded(src);
      // Short clips never reached the pre-buffer threshold.
      if (_autoplay && _handle == null && hasAudio) _startPlayback();
    }
    _streaming = false;
    notifyListeners();
  }

  void _startPlayback() {
    final src = _source;
    if (src == null) return;
    _handle = SoLoud.instance.play(src);
    _paused = false;
    _ticker?.cancel();
    _ticker = Timer.periodic(const Duration(milliseconds: 100), (_) => _tick());
    notifyListeners();
  }

  void _tick() {
    final h = _handle;
    if (h == null) return;
    final s = SoLoud.instance;
    if (!s.isInitialized || !s.getIsValidVoiceHandle(h)) {
      _handle = null;
      _ticker?.cancel();
      position = Duration(milliseconds: (seconds * 1000).round());
      notifyListeners();
      return;
    }
    position = s.getPosition(h);
    notifyListeners();
  }

  /// Plays (or replays) the current clip from the start.
  Future<void> play() async {
    if (_source == null || !hasAudio) return;
    await stop();
    _startPlayback();
  }

  void togglePause() {
    final h = _handle;
    if (h == null) return;
    _paused = !_paused;
    SoLoud.instance.setPause(h, _paused);
    notifyListeners();
  }

  Future<void> stop() async {
    _ticker?.cancel();
    final h = _handle;
    _handle = null;
    _paused = false;
    if (h != null && SoLoud.instance.isInitialized) {
      await SoLoud.instance.stop(h);
    }
    notifyListeners();
  }

  Future<void> _disposeSource() async {
    final src = _source;
    _source = null;
    _streaming = false;
    if (src != null && SoLoud.instance.isInitialized) {
      await SoLoud.instance.disposeSource(src);
    }
  }

  /// The complete clip as one buffer.
  Float32List get clip {
    final out = Float32List(_samples);
    var o = 0;
    for (final c in _chunks) {
      out.setAll(o, c);
      o += c.length;
    }
    return out;
  }

  @override
  void dispose() {
    _ticker?.cancel();
    final h = _handle;
    final src = _source;
    if (SoLoud.instance.isInitialized) {
      if (h != null) SoLoud.instance.stop(h);
      if (src != null) SoLoud.instance.disposeSource(src);
    }
    super.dispose();
  }
}
