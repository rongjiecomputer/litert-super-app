# Overview

Flutter desktop app (Windows/macOS; Android compile-ready) showcasing on-device AI via LiteRT (`flutter_litert`). Currently one use case: 4x image/video super-resolution (Real-ESRGAN). See `README.md` for features and requirements.

## Commands

```sh
flutter pub get
flutter run -d windows            # or -d macos
flutter analyze
flutter test                      # unit tests (tiler, options, paths, ffprobe parsing)
flutter test test/tiler_test.dart # single test file
flutter test integration_test/litert_pipeline_test.dart -d windows \
  --dart-define=SAMPLE_IMAGE=... --dart-define=SAMPLE_VIDEO=... \
  --dart-define=VIDEO_FRAMES=60 --dart-define=OUT_DIR=...   # real LiteRT, needs the model downloaded
```

`test_driver/app_main.dart` is the same app with Flutter Driver enabled.

## Architecture

- **Inference runs in a long-lived background isolate** (`lib/core/inference/inference_worker.dart`) that owns the loaded model. Features send `WorkerTask` objects to it; each task executes next to the model. Cancellation is cooperative (tasks check between tiles). Never touch the model from the UI isolate.
- `model_runner.dart` hides `CompiledModel` (LiteRT 2.x) vs classic `Interpreter` behind one interface; `inference_options.dart` holds options plus per-OS accelerator capabilities.
- Upscale pipeline (`lib/features/upscale/`): `tiler.dart` is pure Dart (split/stitch with overlap); `upscale_tasks.dart` are the worker tasks; `ffmpeg.dart` streams raw frames through two ffmpeg processes (decode → LiteRT → encode, no temp files); `upscale_controller.dart` owns the worker.
- Use cases are registered in `lib/app/use_cases.dart` (entries without a `builder` show as "Planned"); models are declared in `lib/core/models/model_catalog.dart`. README has the add-a-use-case steps.
- Models live in the shared `~/.litert-lm/models/<model-id>/` folder (the `litert-lm` CLI layout); never modify that CLI's `config.json` or `history`. Partial downloads go in `~/.litert-lm/downloading/` as `*.part`.

## Gotchas

- UI uses `package:material_ui` (needs Flutter 3.50+/Dart 3.14).
- Video needs `ffmpeg`/`ffprobe` on PATH or set in Settings.
- Windows: `windows/CMakeLists.txt` works around old Windows SDKs (<10.0.20348) failing on the plugin's C11 custom ops; first CMake configure downloads ~50 MB of GPU runtime.
- macOS: App Sandbox is intentionally disabled in `macos/Runner/*.entitlements` so `~/.litert-lm` and system ffmpeg are reachable.
- `test-image.PNG` / `test-video.mp4` in the repo root are untracked local samples.
