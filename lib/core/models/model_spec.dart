/// Memory layout of a 4-D image tensor.
enum TensorLayout { nhwc, nchw }

/// Static description of a downloadable LiteRT model.
///
/// CompiledModel does not expose tensor shapes, so they are declared here
/// and validated against the runtime's reported byte sizes at load time.
class ModelSpec {
  const ModelSpec({
    required this.id,
    required this.name,
    required this.description,
    required this.hfRepo,
    required this.fileName,
    required this.approxBytes,
    required this.inputShape,
    required this.outputShape,
    required this.inputLayout,
    required this.outputLayout,
    this.license,
  });

  /// Folder name under `~/.litert-lm/models/`.
  final String id;
  final String name;
  final String description;

  /// Hugging Face repository, e.g. `litert-community/real-esrgan-x4v3-litert`.
  final String hfRepo;
  final String fileName;
  final int approxBytes;
  final List<int> inputShape;
  final List<int> outputShape;
  final TensorLayout inputLayout;
  final TensorLayout outputLayout;
  final String? license;

  Uri get downloadUrl =>
      Uri.parse('https://huggingface.co/$hfRepo/resolve/main/$fileName');
  Uri get pageUrl => Uri.parse('https://huggingface.co/$hfRepo');

  int get inputElements => inputShape.fold(1, (a, b) => a * b);
  int get outputElements => outputShape.fold(1, (a, b) => a * b);

  /// Spatial size (H, W) of an image tensor for the given layout.
  static (int h, int w, int c) hwc(List<int> shape, TensorLayout layout) =>
      switch (layout) {
        TensorLayout.nhwc => (shape[1], shape[2], shape[3]),
        TensorLayout.nchw => (shape[2], shape[3], shape[1]),
      };
}
