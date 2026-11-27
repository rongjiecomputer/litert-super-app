import 'dart:io';

import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_selector/file_selector.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'compare_view.dart';
import 'upscale_controller.dart';
import 'workspace_common.dart';

const upscaleImageExtensions = [
  'png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tif', 'tiff', //
];

const _imageTypes = XTypeGroup(
  label: 'Images',
  extensions: upscaleImageExtensions,
  mimeTypes: ['image/*'],
  uniformTypeIdentifiers: ['public.image'],
);

class ImageWorkspace extends StatefulWidget {
  const ImageWorkspace({super.key, required this.controller});
  final UpscaleController controller;

  @override
  State<ImageWorkspace> createState() => _ImageWorkspaceState();
}

class _ImageWorkspaceState extends State<ImageWorkspace> {
  bool _dragging = false;

  UpscaleController get c => widget.controller;

  Future<void> _pick() async {
    final f = await openFile(acceptedTypeGroups: const [_imageTypes]);
    if (f != null) c.setImage(f.path);
  }

  Future<void> _save() async {
    final name = '${p.basenameWithoutExtension(c.imagePath!)}_x4.png';
    String? path;
    if (Platform.isAndroid || Platform.isIOS) {
      final dir = await getApplicationDocumentsDirectory();
      path = p.join(dir.path, name);
    } else {
      final loc = await getSaveLocation(
        suggestedName: name,
        initialDirectory: p.dirname(c.imagePath!),
        acceptedTypeGroups: const [
          XTypeGroup(label: 'PNG', extensions: ['png']),
          XTypeGroup(label: 'JPEG', extensions: ['jpg', 'jpeg']),
        ],
      );
      path = loc?.path;
      if (path != null && p.extension(path).isEmpty) path = '$path.png';
    }
    if (path != null) await c.saveImage(path);
  }

  @override
  Widget build(BuildContext context) {
    final busy = c.isBusy;
    final imageBusy = c.busy == UpscaleBusy.image;
    final hasResult = c.outputImage != null;
    final tiler = c.tiler;
    final inSize = c.imageInputSize;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            OutlinedButton.icon(
              onPressed: busy ? null : _pick,
              icon: const Icon(Icons.image_outlined, size: 18),
              label: const Text('Open image'),
            ),
            FilledButton.icon(
              onPressed: busy || c.imagePath == null || !c.modelReady
                  ? null
                  : c.upscaleImage,
              icon: const Icon(Icons.auto_awesome, size: 18),
              label: const Text('Upscale ×4'),
            ),
            if (imageBusy)
              TextButton.icon(
                onPressed: c.cancel,
                icon: const Icon(Icons.stop_circle_outlined, size: 18),
                label: const Text('Cancel'),
              ),
            if (hasResult)
              OutlinedButton.icon(
                onPressed: busy ? null : _save,
                icon: const Icon(Icons.save_alt, size: 18),
                label: const Text('Save as…'),
              ),
            if (inSize != null)
              Text(
                '${inSize.$1}×${inSize.$2} → '
                '${c.imageOutputSize!.$1}×${c.imageOutputSize!.$2} · '
                '${tiler.tilesFor(inSize.$1, inSize.$2)} tiles of '
                '${tiler.tileW}px',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
        const SizedBox(height: 8),
        StatusLine(controller: c),
        const SizedBox(height: 8),
        Expanded(
          child: DropTarget(
            onDragEntered: (_) => setState(() => _dragging = true),
            onDragExited: (_) => setState(() => _dragging = false),
            onDragDone: (d) {
              setState(() => _dragging = false);
              if (d.files.isNotEmpty) c.setImage(d.files.first.path);
            },
            child: hasResult
                ? CompareView(before: c.inputImage!, after: c.outputImage!)
                : c.imagePath != null
                ? PreviewFrame(
                    highlight: _dragging,
                    child: Image.file(
                      File(c.imagePath!),
                      fit: BoxFit.contain,
                      errorBuilder: (_, e, _) =>
                          Center(child: Text('Cannot preview: $e')),
                    ),
                  )
                : DropZone(
                    highlight: _dragging,
                    icon: Icons.add_photo_alternate_outlined,
                    title: 'Drop an image here',
                    subtitle:
                        'or use “Open image”. Output is 4× each side '
                        '(16× pixels), processed in '
                        '${tiler.tileW}×${tiler.tileH} tiles.',
                  ),
          ),
        ),
      ],
    );
  }
}
