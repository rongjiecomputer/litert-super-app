import 'model_spec.dart';

/// Kokoro voice ids the app downloads (raw `[510, 256]` style tables).
abstract final class KokoroVoices {
  static const english = [
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', //
    'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam',
    'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx',
    'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
  ];
}

/// All models known to the app. New use cases add their models here.
abstract final class ModelCatalog {
  static const realEsrganX4v3 = ModelSpec(
    id: 'real-esrgan-x4v3',
    name: 'Real-ESRGAN x4 v3 (general)',
    description:
        '4× super-resolution, FP16 weights, GPU-friendly graph. '
        'Input 128×128 RGB (NHWC, 0..1) → output 512×512 (NCHW, 0..1).',
    hfRepo: 'litert-community/real-esrgan-x4v3-litert',
    fileName: 'realesr_general_x4v3.tflite',
    approxBytes: 3550000,
    inputShape: [1, 128, 128, 3],
    outputShape: [1, 3, 512, 512],
    inputLayout: TensorLayout.nhwc,
    outputLayout: TensorLayout.nchw,
    license: 'BSD-3-Clause (Real-ESRGAN)',
  );

  static const _kokoroHf =
      'https://huggingface.co/litert-community/Kokoro-82M/resolve/main/legacy';
  static const _kokoroVoices =
      'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices';

  /// Kokoro-82M as the three bucketed FP32 LiteRT graphs (the `.litertlm`
  /// bundles at the repo root need the LiteRT-LM omni runtime), plus the
  /// host-side English front end and 28 English voices.
  static final kokoro82m = ModelSpec(
    id: 'kokoro-82m',
    name: 'Kokoro-82M (TTS)',
    description:
        'StyleTTS2 + iSTFTNet text-to-speech, 24 kHz. Bucketed FP32 graphs '
        '(predictor, prosody, vocoder) with host-side G2P, harmonic source '
        'and iSTFT. English (US/GB), 28 voices.',
    hfRepo: 'litert-community/Kokoro-82M',
    fileName: 'kokoro_vocoder.tflite',
    hfPath: 'legacy/kokoro_vocoder.tflite',
    approxBytes: 433189582,
    license: 'Apache-2.0 (weights) · MIT (G2P) · CMUdict',
    extraFiles: const [
      ModelFile('kokoro_predictor.tflite', '$_kokoroHf/kokoro_predictor.tflite', 90741756),
      ModelFile('kokoro_prosody.tflite', '$_kokoroHf/kokoro_prosody.tflite', 37393104),
      ModelFile(
        'dp_g2p_litert.tflite',
        'https://huggingface.co/litert-community/Kokoro-G2P-en-US/resolve/main/dp_g2p_litert.tflite',
        50969672,
      ),
      ModelFile(
        'config.json',
        'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json',
        2351,
      ),
      // CMU Pronouncing Dictionary, normalised by the LiteRT-Models Kokoro
      // sample (pinned commit).
      ModelFile(
        'cmudict.txt',
        'https://raw.githubusercontent.com/john-rocky/LiteRT-Models/'
            'ceae75d6254ff1370c1a55c1e0279e32dbad274d/kokoro/app/src/main/assets/cmudict.txt',
        3329923,
      ),
    ] + [
      for (final v in KokoroVoices.english)
        ModelFile('voices/$v.bin', '$_kokoroVoices/$v.bin', 522240),
    ],
  );

  static const _qwenHf =
      'https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/legacy';

  /// Qwen3-TTS 0.6B Base as raw LiteRT graphs + host tables, driven by a
  /// Dart port of the reference host loop. `speaker_encoder.tflite` (voice
  /// cloning) is optional: it is produced by
  /// tools/tts/export_qwen3_speaker_encoder.py and imported in the app.
  static final qwen3Tts06b = ModelSpec(
    id: 'qwen3-tts-0.6b',
    name: 'Qwen3-TTS 12Hz 0.6B Base (TTS)',
    description:
        'Multilingual TTS (10 languages), 24 kHz. INT4 talker + FP32 MTP + '
        'FP32 codec decoder, host-side tokenizer, sampling and embeddings. '
        'Voice cloning via speaker x-vectors.',
    hfRepo: 'litert-community/Qwen3-TTS-12Hz-0.6B-Base',
    fileName: 'talker_int4.tflite',
    hfPath: 'legacy/talker_int4.tflite',
    approxBytes: 1887781060,
    license: 'Apache-2.0',
    extraFiles: const [
      ModelFile('mtp_fp32.tflite', '$_qwenHf/mtp_fp32.tflite', 440526692),
      ModelFile('codec_decoder_fp32.tflite', '$_qwenHf/codec_decoder_fp32.tflite', 456820324),
      ModelFile('tokenizer.json', '$_qwenHf/tokenizer.json', 11424262),
      ModelFile('tables/codec_embedding_fp32.npy', '$_qwenHf/tables/codec_embedding_fp32.npy', 12583040),
      ModelFile('tables/mtp_embeddings_fp16.npy', '$_qwenHf/tables/mtp_embeddings_fp16.npy', 62914688),
      ModelFile('tables/text_embedding_fp16.npy', '$_qwenHf/tables/text_embedding_fp16.npy', 622329984),
      ModelFile('tables/text_projection_fp32.npz', '$_qwenHf/tables/text_projection_fp32.npz', 25179078),
      ModelFile('voices/demo_speaker.npy', '$_qwenHf/voices/demo_speaker.npy', 4224),
      // Built locally by tools/tts/export_qwen3_speaker_encoder.py.
      ModelFile('speaker_encoder.tflite', '', 0, optional: true),
    ],
  );

  static final all = <ModelSpec>[realEsrganX4v3, kokoro82m, qwen3Tts06b];
}
