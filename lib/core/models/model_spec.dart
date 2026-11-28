/// Memory layout of a 4-D image tensor.
enum TensorLayout { nhwc, nchw }

/// One file of a multi-file model, stored at `models/<id>/<relPath>`.
class ModelFile {
  const ModelFile(this.relPath, this.url, this.approxBytes, {this.optional = false});

  /// Path relative to the model folder (may contain `/`).
  final String relPath;

  /// Absolute download URL (files may come from several repositories).
  final String url;
  final int approxBytes;

  /// Optional files are not required for the model to be "ready" and are
  /// not downloaded automatically (e.g. a user-converted add-on graph).
  final bool optional;

  Uri get uri => Uri.parse(url);
}

/// A file taken out of a downloaded `.litertlm` bundle: the first section of
/// [dataType] (a `LitertLmType` value) whose metadata has [key] = [value].
class BundleEntry {
  const BundleEntry(
    this.relPath,
    this.approxBytes, {
    required this.dataType,
    required this.key,
    required this.value,
  });

  final String relPath;
  final int approxBytes;
  final int dataType;
  final String key;
  final String value;
}

/// A `.litertlm` bundle that is downloaded once and split into plain files
/// (what `litert-lm unpack` does). The bundle itself is not kept.
class ModelBundle {
  const ModelBundle(this.url, this.approxBytes, this.entries);

  final String url;

  /// Size of the whole bundle download.
  final int approxBytes;
  final List<BundleEntry> entries;

  Uri get uri => Uri.parse(url);
}

/// Static description of a downloadable LiteRT model.
///
/// Tensor shapes are declared here (they also document the model) and
/// validated against the compiled graph's reported tensors at load time.
/// Non-image models leave the shape/layout fields at their defaults and list
/// their additional files in [extraFiles].
class ModelSpec {
  const ModelSpec({
    required this.id,
    required this.name,
    required this.description,
    required this.hfRepo,
    required this.fileName,
    this.hfPath,
    required this.approxBytes,
    this.inputShape = const [],
    this.outputShape = const [],
    this.inputLayout = TensorLayout.nhwc,
    this.outputLayout = TensorLayout.nhwc,
    this.extraFiles = const [],
    this.bundle,
    this.license,
  });

  /// Folder name under `~/.litert-lm/models/`.
  final String id;
  final String name;
  final String description;

  /// Hugging Face repository, e.g. `litert-community/real-esrgan-x4v3-litert`.
  final String hfRepo;
  /// Primary file, downloaded from [hfRepo].
  final String fileName;

  /// Path of [fileName] inside [hfRepo] when it differs (e.g. `legacy/x`).
  final String? hfPath;

  /// Approximate total size of all required files.
  final int approxBytes;
  final List<int> inputShape;
  final List<int> outputShape;
  final TensorLayout inputLayout;
  final TensorLayout outputLayout;
  final String? license;

  /// Additional files beyond [fileName].
  final List<ModelFile> extraFiles;

  /// When set, [fileName] and the model's main files come from this bundle
  /// instead of being downloaded individually; [approxBytes] then covers the
  /// bundle plus [extraFiles].
  final ModelBundle? bundle;

  Uri get downloadUrl => Uri.parse(
    'https://huggingface.co/$hfRepo/resolve/main/${hfPath ?? fileName}',
  );
  /// Every file of the model, primary first.
  List<ModelFile> get files => [
    if (bundle case final b?)
      for (final e in b.entries) ModelFile(e.relPath, b.url, e.approxBytes)
    else
      ModelFile(fileName, downloadUrl.toString(), approxBytes - _extraBytes),
    ...extraFiles,
  ];

  /// The bundle entry that produces [relPath], if any.
  BundleEntry? bundleEntry(String relPath) {
    for (final e in bundle?.entries ?? const <BundleEntry>[]) {
      if (e.relPath == relPath) return e;
    }
    return null;
  }

  /// Files that must exist for the model to be usable.
  List<ModelFile> get requiredFiles =>
      files.where((f) => !f.optional).toList(growable: false);

  int get _extraBytes => extraFiles
      .where((f) => !f.optional)
      .fold(0, (a, f) => a + f.approxBytes);

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
