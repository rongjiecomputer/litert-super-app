import 'model_spec.dart';

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

  static const all = <ModelSpec>[realEsrganX4v3];
}
