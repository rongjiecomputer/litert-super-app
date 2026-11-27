import 'package:material_ui/material_ui.dart';
import 'package:path/path.dart' as p;

import '../../app/services.dart';
import '../../core/inference/inference_options.dart';
import '../../core/metrics/metrics_panel.dart';
import '../../core/widgets/inference_options_panel.dart';
import '../../core/widgets/model_banner.dart';
import '../../core/widgets/page_header.dart';
import '../../core/widgets/section_card.dart';
import 'image_workspace.dart';
import 'upscale_controller.dart';
import 'video_workspace.dart';

enum _Mode { image, video }

class UpscalePage extends StatefulWidget {
  const UpscalePage({super.key});

  @override
  State<UpscalePage> createState() => _UpscalePageState();
}

class _UpscalePageState extends State<UpscalePage> {
  late final UpscaleController c;
  _Mode _mode = _Mode.image;

  @override
  void initState() {
    super.initState();
    final s = AppServices.instance;
    c = UpscaleController(
      settings: s.settings,
      models: s.models,
      caps: PlatformCaps.current(),
    );
    final file = s.launchFile;
    if (file != null) {
      final ext = p.extension(file).toLowerCase().replaceFirst('.', '');
      if (upscaleImageExtensions.contains(ext)) {
        s.launchFile = null;
        c.setImage(file);
      } else if (upscaleVideoExtensions.contains(ext) && c.caps.supportsVideo) {
        s.launchFile = null;
        _mode = _Mode.video;
        c.setVideo(file);
      }
    }
  }

  @override
  void dispose() {
    c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([c, c.models]),
      builder: (context, _) => LayoutBuilder(
        builder: (context, box) {
          final wide = box.maxWidth >= 1000;
          final workspace = _workspace();
          final side = _sidePanel();
          final header = Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              PageHeader(
                icon: Icons.photo_size_select_large,
                title: 'Super-resolution',
                subtitle:
                    '${c.spec.name} · 4× image & video upscaling, fully '
                    'on-device with LiteRT',
              ),
              ModelBanner(manager: c.models, spec: c.spec),
            ],
          );
          if (wide) {
            return Padding(
              padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  header,
                  Expanded(
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Expanded(child: workspace),
                        const SizedBox(width: 20),
                        SizedBox(
                          width: 380,
                          child: SingleChildScrollView(child: side),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          }
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              header,
              SizedBox(height: 560, child: workspace),
              const SizedBox(height: 16),
              side,
            ],
          );
        },
      ),
    );
  }

  Widget _workspace() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: SegmentedButton<_Mode>(
            segments: const [
              ButtonSegment(
                value: _Mode.image,
                icon: Icon(Icons.image_outlined),
                label: Text('Image'),
              ),
              ButtonSegment(
                value: _Mode.video,
                icon: Icon(Icons.movie_outlined),
                label: Text('Video'),
              ),
            ],
            selected: {_mode},
            onSelectionChanged: c.isBusy
                ? null
                : (s) => setState(() => _mode = s.first),
          ),
        ),
        const SizedBox(height: 12),
        Expanded(
          child: switch (_mode) {
            _Mode.image => ImageWorkspace(controller: c),
            _Mode.video => VideoWorkspace(controller: c),
          },
        ),
      ],
    );
  }

  Widget _sidePanel() {
    final enabled = !c.isBusy;
    final t = c.tiler;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        InferenceOptionsPanel(
          options: c.options,
          caps: c.caps,
          enabled: enabled,
          loadInfo: c.loadInfo,
          needsReload: c.needsReload,
          onChanged: c.setOptions,
          onLoad: c.modelReady ? c.loadModel : null,
          onUnload: c.unloadModel,
          extra: [
            LabeledField(
              label: 'Tile overlap: ${c.overlap}px',
              help:
                  'Context pixels added around each ${t.tileW}px tile and '
                  'cropped after upscaling, which hides seams. More overlap '
                  'means more tiles.',
              child: Slider(
                value: c.overlap.toDouble(),
                min: 0,
                max: 32,
                divisions: 8,
                label: '${c.overlap}px',
                onChanged: enabled
                    ? (v) => c.setParams(overlap: v.round())
                    : null,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        MetricsPanel(
          loadInfo: c.loadInfo,
          lastRun: c.lastRun,
          history: c.history,
        ),
      ],
    );
  }
}
