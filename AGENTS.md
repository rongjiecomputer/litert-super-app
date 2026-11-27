# Overview

Flutter desktop app (Windows/macOS; Android compile-ready) showcasing on-device AI via LiteRT (`flutter_litert`). Use cases: 4x image/video super-resolution (Real-ESRGAN) and text to speech (Kokoro-82M, Qwen3-TTS 0.6B with voice cloning). See `README.md` for features and requirements.

## Commands

```sh
flutter pub get
flutter run -d windows            # or -d macos
flutter analyze
flutter test                      # unit tests (tiler, options, paths, ffprobe, TTS dsp/tokenizer/g2p)
flutter test test/tiler_test.dart # single test file
flutter test integration_test/litert_pipeline_test.dart -d windows \
  --dart-define=SAMPLE_IMAGE=... --dart-define=SAMPLE_VIDEO=... \
  --dart-define=VIDEO_FRAMES=60 --dart-define=OUT_DIR=...   # real LiteRT, needs the model downloaded
```

`integration_test/tts_pipeline_test.dart` runs both TTS engines on real LiteRT (`--dart-define=TTS_ENGINES=kokoro,qwen3 --dart-define=OUT_DIR=...`). `tools/tts/` has the graph inspector, golden generator (`py -I tools/tts/make_goldens.py`) and the speaker-encoder exporter.

`test_driver/app_main.dart` is the same app with Flutter Driver enabled.

## Architecture

- **Inference runs in a long-lived background isolate** (`lib/core/inference/inference_worker.dart`) that owns the loaded model. Features send `WorkerTask` objects to it; each task executes next to the model. Cancellation is cooperative (tasks check between tiles). Never touch the model from the UI isolate.
- `model_runner.dart` hides `CompiledModel` (LiteRT 2.x) vs classic `Interpreter` behind one interface; `inference_options.dart` holds options plus per-OS accelerator capabilities.
- Upscale pipeline (`lib/features/upscale/`): `tiler.dart` is pure Dart (split/stitch with overlap); `upscale_tasks.dart` are the worker tasks; `ffmpeg.dart` streams raw frames through two ffmpeg processes (decode → LiteRT → encode, no temp files); `upscale_controller.dart` owns the worker.
- TTS (`lib/features/tts/`): multi-graph pipelines implement `WorkerEngine` and are loaded into `ctx.engine` (not `ModelRunner`). They use the legacy `.tflite` graphs via `Interpreter` + `SignatureRunner` (`engines/graph.dart`), with zero-copy tensor views. Host-side DSP/text code is pure Dart and unit-tested against goldens in `test/goldens/tts/`. Audio streams to the UI as `TtsAudioChunk` progress messages and plays through `flutter_soloud`.
- Use cases are registered in `lib/app/use_cases.dart` (entries without a `builder` show as "Planned"); models are declared in `lib/core/models/model_catalog.dart`. README has the add-a-use-case steps.
- Models live in the shared `~/.litert-lm/models/<model-id>/` folder (the `litert-lm` CLI layout); never modify that CLI's `config.json` or `history`. Partial downloads go in `~/.litert-lm/downloading/` as `*.part`.

## Gotchas

- UI uses `package:material_ui` (needs Flutter 3.50+/Dart 3.14).
- Video needs `ffmpeg`/`ffprobe` on PATH or set in Settings.
- TTS is Interpreter-only: `CompiledModel` is float32 and signature-0 only. The `.litertlm` TTS bundles need LiteRT-LM omni, which has no Windows build or Dart bindings, so the app uses each HF repo's `legacy/` graphs.
- Zero all graph inputs after `allocateTensors` (`Sig` does this). Garbage arena memory made the Kokoro graphs fail, or run about 20× slower.
- Kokoro: the `istft_W*.bin` files in its legacy folder don't reproduce the reference audio; `KokoroDsp.istft` is the torch-style iSTFT, verified against the ONNX export.
- Qwen3 voice cloning needs `speaker_encoder.tflite`, built by `tools/tts/export_qwen3_speaker_encoder.py`. It needs `litert-torch`, which has no Windows wheels, so run it on Linux/macOS/WSL.
- macOS mic access needs `NSMicrophoneUsageDescription` and the `com.apple.security.device.audio-input` entitlement (both set).
- Windows: `windows/CMakeLists.txt` works around old Windows SDKs (<10.0.20348) failing on the plugin's C11 custom ops; first CMake configure downloads ~50 MB of GPU runtime.
- macOS: App Sandbox is intentionally disabled in `macos/Runner/*.entitlements` so `~/.litert-lm` and system ffmpeg are reachable.
- `test-image.PNG` / `test-video.mp4` in the repo root are untracked local samples.
