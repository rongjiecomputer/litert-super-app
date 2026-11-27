# LiteRT Super App

A Flutter desktop app (Windows, macOS; Android compile-ready) that showcases
on-device AI use cases running on [LiteRT](https://ai.google.dev/edge/litert).
Everything runs through the LiteRT **`CompiledModel` C API**, called from Dart
FFI against the prebuilt `libLiteRt` runtime (no `flutter_litert` plugin, no
`Interpreter`).

**Use case 1 — Super-resolution:** 4× image and video upscaling with
[Real-ESRGAN x4 v3](https://huggingface.co/litert-community/real-esrgan-x4v3-litert).

**Use case 2 — Text to speech:** streamed speech synthesis with
[Kokoro-82M](https://huggingface.co/litert-community/Kokoro-82M) (English,
28 voices) and
[Qwen3-TTS 12Hz 0.6B Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base)
(10 languages, on-device voice cloning).

## Features

- **Left navigation rail** on desktop (extended labels ≥ 1280 px). Phones and
  narrow windows get a bottom navigation bar instead.
- **Per-use-case inference options**, saved between sessions:
  - Accelerator: CPU / GPU / NPU, filtered to what the current OS supports.
  - Precision: FP32 / FP16 (GPU only).
  - Run unsupported ops on the CPU, CPU thread count, and tile overlap.
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
- **Text to speech:** type text, pick an engine and a voice, and audio plays
  while it is still being generated. You can save the result as WAV. Qwen3-TTS
  can clone a voice from a short reference clip: drop or pick an audio file,
  or record one from the microphone. Cloned voices are saved for reuse. See
  [Text to speech](#text-to-speech) below.
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
  models/kokoro-82m/…, models/qwen3-tts-0.6b/…          ← this app (multi-file)
  models/gemma4-e2b/model.litertlm                      ← litert-lm CLI (listed, untouched)
  downloading/                                          ← partial downloads (*.part)
```

The app never modifies the CLI's `config.json` or `history`. On Android, the
same layout lives under the app's support directory.

## Text to speech

Both models ship as `.litertlm` bundles meant for LiteRT-LM's experimental
C++ "omni" runtime. That runtime has no Windows build and no Dart bindings,
so the app uses the raw `.tflite` graphs from each repo's `legacy/` folder
instead. It drives them through `CompiledModel` by signature key, with int32 /
int64 inputs and per-graph signatures (`prefill` / `decode`).

**GPU placement is per graph.** With GPU selected, only graphs where it pays
off run there; the rest stay on CPU. Measured on a GTX 1050 (WebGPU over
Direct3D 12):

| Graph | CPU (8T) | GPU | Placement |
|---|---|---|---|
| Real-ESRGAN tile (upscaling) | 589 ms | 44 ms | GPU |
| Qwen3 codec decoder | ~5 s | ~0.6 s | GPU |
| Qwen3 talker decode / MTP | 85 / 32 ms | 215 / 35 ms | CPU |
| Kokoro predictor / prosody | 0.4 / 0.26 s | 3.8 / 2.4 s, and the two together exhaust 2 GB VRAM | CPU |
| Kokoro vocoder | 9 s | invalid shaders on this adapter (silent output) | CPU |

Other GPUs will differ. The WebGPU runtime reports shader validation errors
only in its log and then returns garbage rather than failing, so check
accelerator output (the PSNR test in `integration_test/`) when trying a new
adapter. GPU FP16 needs the adapter's `shader-f16` feature; the app detects
its absence and recompiles in FP32.

| Engine | Download | What runs on the host (Dart) |
|---|---|---|
| Kokoro-82M | ~433 MB: predictor, prosody, vocoder (FP32), neural G2P, CMUdict, 28 voices | Text normalisation, CMUdict and neural G2P to misaki IPA, duration rounding and alignment, hn-NSF harmonic source + STFT, iSTFT |
| Qwen3-TTS 0.6B | ~1.9 GB: INT4 talker, FP32 MTP, FP32 codec decoder, embedding tables, tokenizer | Qwen2 BPE tokenizer, text-embedding lookup and projection MLP, KV-cache handling, top-k/temperature sampling, codec windowing |

- **Kokoro** is English-only on this path (US and GB voices); the
  multilingual front end is part of the omni runtime. Each chunk runs a fixed
  12.8 s bucket, so short texts cost as much as long ones.
- **Qwen3-TTS** is a port of the reference host loop
  (`qwen3_tts_pipeline.py` in litert-samples). Audio starts after the first
  16 codec frames and then streams in 3 s windows. On CPU it is usually slower
  than real time, so playback pauses to buffer.
- **Voice cloning (Qwen3-TTS Base).** The talker is conditioned on a 1024-d
  speaker x-vector. Only this x-vector mode is supported; cloning from a
  reference transcript (ICL) needs the codec encoder, which is not converted.
  The LiteRT repo does not ship the speaker encoder, so build it once:

  ```sh
  # Linux/macOS (on Windows: inside WSL). litert-torch has no Windows wheels.
  python3 -m venv ~/litert-venv
  ~/litert-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch<2.15'
  ~/litert-venv/bin/pip install litert-torch safetensors librosa numpy requests
  ~/litert-venv/bin/python tools/tts/export_qwen3_speaker_encoder.py --out speaker_encoder.tflite
  ```

  The script downloads only the encoder weights (about 18 MB, using HTTP range
  requests on `Qwen/Qwen3-TTS-12Hz-0.6B-Base`). It exports fixed 3/6/10 s
  signatures and checks the result against PyTorch. Import the file from the
  "Clone a voice" card, or copy it to
  `~/.litert-lm/models/qwen3-tts-0.6b/speaker_encoder.tflite`.
  Cloned voices are stored as `.npy` files in
  `models/qwen3-tts-0.6b/voices/cloned/`.

## Requirements

- Flutter 3.50+ (Dart 3.14). The UI uses `package:material_ui`.
- **LiteRT runtime and headers (once per clone).** They are not in git:

  ```sh
  py -I tools/fetch_litert_sdk.py                       # headers + this OS's runtime
  py -I tools/fetch_litert_sdk.py --platform android_arm64   # also Android
  ```

  The script reads a pinned snapshot of Google's `litert_prebuilts.zip` using
  HTTP range requests, so it downloads about 40 MB instead of the 360 MB
  archive. It fills `litert_cc_sdk/` (C/C++ headers) and
  `litert_runtime/<platform>/` (`libLiteRt`, the GPU accelerator, and on
  Windows `dxcompiler.dll` / `dxil.dll` from the Windows SDK or the
  DirectXShaderCompiler release). The platform builds bundle those libraries
  into the app: `windows/CMakeLists.txt` (next to the `.exe`), an Xcode build
  phase on macOS (`Contents/Frameworks`) and `jniLibs` on Android.
- **Video** (and non-WAV voice-cloning references): `ffmpeg` and `ffprobe`
  must be on PATH, or set in Settings.
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - The app also checks WinGet Links, Scoop, Chocolatey and the Homebrew folders.
- **Windows:** x64. GPU inference runs on WebGPU over Direct3D 12; the first
  run of a model compiles its shaders and can take tens of seconds.
- **macOS:** Apple Silicon only (the prebuilt runtime has no x86_64 build).
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
flutter test                  # tiler, options, paths, ffprobe, TTS DSP/tokenizer/G2P

# Real LiteRT on the host: CPU / GPU configs + PSNR vs CPU + ffmpeg video
flutter test integration_test/litert_pipeline_test.dart -d windows \
  --dart-define=SAMPLE_IMAGE=path/to/image.png \
  --dart-define=SAMPLE_VIDEO=path/to/clip.mp4 \
  --dart-define=VIDEO_FRAMES=60 --dart-define=OUT_DIR=path/to/out
```

```sh
# TTS engines on real LiteRT (needs the models; writes WAVs to OUT_DIR)
flutter test integration_test/tts_pipeline_test.dart -d windows \
  --dart-define=OUT_DIR=path/to/out [--dart-define=TTS_ENGINES=kokoro,qwen3] \
  [--dart-define=TTS_ACCEL=gpu]

# Compile + run any .tflite and print its signatures / accelerator / timings
$env:LITERT_SMOKE_MODEL = "path/to/model.tflite"; $env:LITERT_SMOKE_ARGS = "--gpu"
flutter test test/litert_smoke_test.dart

# Regenerate TTS goldens (tokenizer ids, mel reference); needs tokenizers + librosa
py -I tools/tts/make_goldens.py
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
    litert/
      ffi.dart                  dart:ffi bindings to the libLiteRt C API
      compiled_graph.dart       CompiledModel: signatures, tensors, buffers, options
    inference/
      inference_options.dart    options + per-OS capabilities
      model_runner.dart         single float-in/float-out runner (upscaling)
      inference_worker.dart     long-lived isolate owning the model; runs WorkerTasks
    metrics/                    RunMetrics, RSS sampler, metrics panel
    widgets/                    options panel, model banner, section cards
  features/
    upscale/      tiler (pure Dart), worker tasks, ffmpeg pipeline, controller, UI
    tts/
      engines/    Kokoro + Qwen3-TTS pipelines (WorkerEngine), speaker encoder, sampling
      dsp/        FFT, mel, Kokoro harmonic source/iSTFT, WAV, resampling
      text/       Qwen BPE tokenizer, English G2P, .npy/.npz reader
      …           worker tasks, controller, streaming player, recorder, UI
tools/fetch_litert_sdk.py   downloads the LiteRT headers + prebuilt runtime
tools/tts/        graph inspector, golden generator, speaker-encoder exporter
    home/ models/ settings/
```

Inference runs in a background isolate, so the UI stays responsive. A use
case sends `WorkerTask` objects to that isolate, and each task runs next to
the loaded model there. Cancellation is cooperative: tasks check for it
between tiles (or between frames and chunks for TTS). Use cases whose models
don't fit the single float32 input/output `ModelRunner`, such as the
multi-graph TTS pipelines, load a `WorkerEngine` into the worker instead
(`ctx.engine`).

### Adding a use case

1. Add its `ModelSpec` to `lib/core/models/model_catalog.dart`. Models made
   of several files list them in `extraFiles`, using full URLs.
2. Create `lib/features/<name>/` with:
   - `WorkerTask`s for pre/post-processing around `ctx.requireRunner().run(...)`.
   - A controller that owns an `InferenceWorker`.
   - A page that reuses `InferenceOptionsPanel` and `MetricsPanel`.
3. Register it in `lib/app/use_cases.dart` by setting a `builder`. Entries
   without a builder show as "Planned" on Home.
