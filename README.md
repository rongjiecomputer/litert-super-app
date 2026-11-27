# LiteRT Super App

A Flutter desktop app (Windows, macOS; Android compile-ready) that showcases
on-device AI use cases running on [LiteRT](https://ai.google.dev/edge/litert)
through the official
[`flutter_litert`](https://github.com/google-ai-edge/flutter_litert) plugin.

**Use case 1 — Super-resolution:** 4× image and video upscaling with
[Real-ESRGAN x4 v3](https://huggingface.co/litert-community/real-esrgan-x4v3-litert).

## Features

- **Left navigation rail** on desktop (extended labels ≥ 1280 px). Phones and
  narrow windows get a bottom navigation bar instead.
- **Per-use-case inference options**, saved between sessions:
  - Runtime: `CompiledModel` (LiteRT 2.x) or the classic `Interpreter`.
  - Accelerator: CPU / GPU / NPU, filtered to what the current OS supports.
  - Precision: FP32 / FP16. FP16 applies to the GPU, or to XNNPACK force-FP16 on CPU.
  - Tensor buffers: managed or host memory.
  - Fall back to CPU if compilation fails, CPU thread count, and tile overlap.
- **Inference cost** for each run:
  - Model load/compile time, warm-up run, RAM at load.
  - Per-tile latency (avg / p50 / p90 / min / max) and throughput.
  - Decode, pre-process and post-process time.
  - RAM before / peak / after, and the process max RSS.
  - Which accelerators actually ran, including any CPU fallback.
  - A run history for comparing configurations.
- **Image:** open a file or drag and drop it, then compare before and after with a zoomable split view. Save as PNG or JPEG.
- **Video:** ffmpeg decodes raw frames, LiteRT upscales each one, and a second
  ffmpeg encodes H.264 and re-encodes the original audio. No temporary files
  are written. The view shows a live frame preview, fps and ETA, and you can
  cancel at any point.
- **Models page:** download, delete, or show a model in its folder. It also
  lists other models found in the shared folder, such as LiteRT-LM bundles.
- **Open with:** pass a file path on the command line
  (`ai_litert.exe photo.png`) to open it straight in the upscaler.

## Model storage

Desktop models live in `~/.litert-lm/models/<model-id>/`, the layout the
`litert-lm` CLI uses:

```
~/.litert-lm/
  models/real-esrgan-x4v3/realesr_general_x4v3.tflite   ← this app
  models/gemma4-e2b/model.litertlm                      ← litert-lm CLI (listed, untouched)
  downloading/                                          ← partial downloads (*.part)
```

The app never modifies the CLI's `config.json` or `history`. On Android, the
same layout lives under the app's support directory.

## Requirements

- Flutter 3.50+ (Dart 3.14). The UI uses `package:material_ui`.
- **Video:** `ffmpeg` and `ffprobe` must be on PATH, or set in Settings.
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - The app also checks WinGet Links, Scoop, Chocolatey and the Homebrew folders.
- **Windows:** The first CMake configure downloads `flutter_litert`'s GPU
  runtime (about 50 MB: `libLiteRt.dll`, WebGPU accelerator, DXC). GPU
  inference runs on WebGPU over Direct3D 12.
  - Windows SDKs older than 10.0.20348 can't compile the plugin's C11 custom
    ops. `windows/CMakeLists.txt` works around this automatically. Installing
    a newer SDK through the Visual Studio Installer is the cleaner fix.
- **macOS:** App Sandbox is disabled in `macos/Runner/*.entitlements`. A
  sandboxed `HOME` would point into the app container, which hides the shared
  `~/.litert-lm` folder and the system ffmpeg. GPU runs on Metal and NPU on
  Core ML (Apple Silicon, macOS 14+).

## Run

```sh
flutter pub get
flutter run -d windows        # or -d macos
```

## Tests

```sh
flutter analyze
flutter test                  # tiler stitching, options, paths, ffprobe parsing

# Real LiteRT on the host: 5 runtime/accelerator configs + PSNR vs CPU + ffmpeg video
flutter test integration_test/litert_pipeline_test.dart -d windows \
  --dart-define=SAMPLE_IMAGE=path/to/image.png \
  --dart-define=SAMPLE_VIDEO=path/to/clip.mp4 \
  --dart-define=VIDEO_FRAMES=60 --dart-define=OUT_DIR=path/to/out
```

`test_driver/app_main.dart` is the same app with Flutter Driver enabled, for
automated UI checks.

## Architecture

```
lib/
  app/            shell (nav rail / bar), use-case registry, services
  core/
    paths.dart                  ~/.litert-lm resolution
    models/                     ModelSpec, catalog, download manager
    inference/
      inference_options.dart    options + per-OS capabilities
      model_runner.dart         CompiledModel / Interpreter behind one interface
      inference_worker.dart     long-lived isolate owning the model; runs WorkerTasks
    metrics/                    RunMetrics, RSS sampler, metrics panel
    widgets/                    options panel, model banner, section cards
  features/
    upscale/      tiler (pure Dart), worker tasks, ffmpeg pipeline, controller, UI
    home/ models/ settings/
```

Inference runs in a background isolate, so the UI stays responsive. A use
case sends `WorkerTask` objects to that isolate, and each task runs next to
the loaded model there. Cancellation is cooperative: tasks check for it
between tiles.

### Adding a use case

1. Add its `ModelSpec` to `lib/core/models/model_catalog.dart`.
2. Create `lib/features/<name>/` with:
   - `WorkerTask`s for pre/post-processing around `ctx.requireRunner().run(...)`.
   - A controller that owns an `InferenceWorker`.
   - A page that reuses `InferenceOptionsPanel` and `MetricsPanel`.
3. Register it in `lib/app/use_cases.dart` by setting a `builder`. Entries
   without a builder show as "Planned" on Home.
