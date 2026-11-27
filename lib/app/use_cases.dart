import 'package:material_ui/material_ui.dart';

import '../core/models/model_catalog.dart';
import '../core/models/model_spec.dart';
import '../features/upscale/image_workspace.dart';
import '../features/upscale/upscale_page.dart';
import '../features/upscale/video_workspace.dart';

/// One showcase entry in the super app.
///
/// To add a use case: add its models to [ModelCatalog], create a
/// `features/<name>/` folder with a page (and `WorkerTask`s for the
/// inference worker), then register it here.
class UseCase {
  const UseCase({
    required this.id,
    required this.title,
    required this.summary,
    required this.icon,
    required this.selectedIcon,
    this.models = const [],
    this.tags = const [],
    this.opens = const [],
    this.builder,
  });

  final String id;
  final String title;
  final String summary;
  final IconData icon;
  final IconData selectedIcon;
  final List<ModelSpec> models;
  final List<String> tags;

  /// File extensions this use case can open from the command line.
  final List<String> opens;

  /// Null = planned (shown on Home but not in navigation).
  final WidgetBuilder? builder;

  bool get available => builder != null;
}

final useCases = <UseCase>[
  UseCase(
    id: 'upscale',
    title: 'Upscaling',
    summary:
        'Real-ESRGAN 4× super-resolution for images and videos, '
        'tiled 128px inference on CPU, GPU or NPU.',
    icon: Icons.photo_size_select_large_outlined,
    selectedIcon: Icons.photo_size_select_large,
    models: const [ModelCatalog.realEsrganX4v3],
    tags: const ['Vision', 'Image', 'Video'],
    opens: const [...upscaleImageExtensions, ...upscaleVideoExtensions],
    builder: (_) => const UpscalePage(),
  ),
  const UseCase(
    id: 'segmentation',
    title: 'Segmentation',
    summary: 'Background removal and per-pixel masks.',
    icon: Icons.layers_outlined,
    selectedIcon: Icons.layers,
    tags: ['Vision'],
  ),
  const UseCase(
    id: 'detection',
    title: 'Object detection',
    summary: 'Real-time boxes and labels from camera or files.',
    icon: Icons.center_focus_strong_outlined,
    selectedIcon: Icons.center_focus_strong,
    tags: ['Vision'],
  ),
  const UseCase(
    id: 'llm',
    title: 'LLM chat',
    summary: 'Gemma via LiteRT-LM, sharing ~/.litert-lm/models.',
    icon: Icons.chat_bubble_outline,
    selectedIcon: Icons.chat_bubble,
    tags: ['Language'],
  ),
];
