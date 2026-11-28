import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import '../../core/inference/inference_options.dart';
import '../../core/inference/inference_worker.dart';
import '../../core/metrics/run_metrics.dart';
import 'dsp/audio.dart';
import 'engines/kokoro_engine.dart';
import 'engines/qwen3_tts_engine.dart';
import 'engines/speaker_encoder.dart';
import 'engines/tts_engine.dart';

/// Result of loading a TTS engine into the worker.
class TtsLoadResult {
  const TtsLoadResult(this.info, this.voices);
  final LoadInfo info;
  final List<TtsVoice> voices;
}

/// Loads [kind] from [modelDir] (closing whatever was resident) and warms
/// every graph up.
class LoadTtsEngineTask extends WorkerTask<TtsLoadResult> {
  const LoadTtsEngineTask(this.kind, this.modelDir, this.options);

  final TtsEngineKind kind;
  final String modelDir;
  final InferenceOptions options;

  @override
  Future<TtsLoadResult> run(WorkerContext ctx) async {
    ctx.unloadAll();
    final rssBefore = ProcessInfo.currentRss;
    final sw = Stopwatch()..start();
    final TtsEngine engine = switch (kind) {
      TtsEngineKind.kokoro => KokoroEngine.load(modelDir, options),
      TtsEngineKind.qwen3 => Qwen3TtsEngine.load(modelDir, options),
    };
    final loadMs = sw.elapsedMicroseconds / 1000;
    ctx.engine = engine;
    sw
      ..reset()
      ..start();
    try {
      await engine.warmUp(ctx);
    } catch (_) {
      ctx.unloadAll();
      rethrow;
    }
    return TtsLoadResult(
      LoadInfo(
        optionsSummary: options.summary,
        runtimeLabel: '${kind.label} · LiteRT CompiledModel',
        requested: [engine.accelerator],
        actual: [engine.accelerator],
        didFallback: engine.fallbackReason != null,
        fallbackReason: engine.fallbackReason,
        fullyAccelerated: null,
        loadMs: loadMs,
        warmupMs: sw.elapsedMicroseconds / 1000,
        rssDeltaBytes: ProcessInfo.currentRss - rssBefore,
      ),
      engine.voices,
    );
  }
}

/// One chunk of synthesized audio, sent as progress while synthesizing.
class TtsAudioChunk {
  TtsAudioChunk(Float32List pcm)
    : _data = TransferableTypedData.fromList([pcm]),
      samples = pcm.length;

  final TransferableTypedData _data;
  final int samples;

  Float32List materialize() => _data.materialize().asFloat32List();
}

/// Synthesizes [request] with the resident engine, streaming audio chunks
/// through progress messages.
class SynthesizeTask extends WorkerTask<TtsStats> {
  const SynthesizeTask(this.request);
  final TtsRequest request;

  @override
  Future<TtsStats> run(WorkerContext ctx) {
    final engine = ctx.requireEngine<TtsEngine>();
    return engine.synthesize(
      request,
      ctx,
      (pcm) => ctx.progress(TtsAudioChunk(pcm)),
    );
  }
}

/// Computes a Qwen3-TTS speaker x-vector from reference audio with the
/// exported speaker encoder. Runs next to the TTS engine (the encoder is
/// small and loaded per call).
class CloneVoiceTask extends WorkerTask<Float32List> {
  const CloneVoiceTask(this.encoderPath, this.options, this.audio, this.sampleRate);

  final String encoderPath;
  final InferenceOptions options;
  final Float32List audio;
  final int sampleRate;

  @override
  Future<Float32List> run(WorkerContext ctx) async {
    final enc = SpeakerEncoder.load(encoderPath, options);
    try {
      return enc.embed(PcmAudio(audio, sampleRate));
    } finally {
      enc.close();
    }
  }
}
