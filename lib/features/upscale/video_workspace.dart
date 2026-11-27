import 'dart:io';

import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_selector/file_selector.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path/path.dart' as p;

import '../../core/system.dart';
import 'compare_view.dart';
import 'upscale_controller.dart';
import 'workspace_common.dart';

const upscaleVideoExtensions = ['mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v'];

const _videoTypes = XTypeGroup(
  label: 'Videos',
  extensions: upscaleVideoExtensions,
  mimeTypes: ['video/*'],
  uniformTypeIdentifiers: ['public.movie'],
);

class VideoWorkspace extends StatefulWidget {
  const VideoWorkspace({super.key, required this.controller});
  final UpscaleController controller;

  @override
  State<VideoWorkspace> createState() => _VideoWorkspaceState();
}

class _VideoWorkspaceState extends State<VideoWorkspace> {
  bool _dragging = false;
  UpscaleController get c => widget.controller;

  @override
  void initState() {
    super.initState();
    if (!c.ffmpegChecked && c.caps.supportsVideo) c.detectFfmpeg();
  }

  Future<void> _pick() async {
    final f = await openFile(acceptedTypeGroups: const [_videoTypes]);
    if (f != null) await c.setVideo(f.path);
  }

  Future<void> _start() async {
    final suggested = c.defaultVideoOutput();
    final loc = await getSaveLocation(
      suggestedName: p.basename(suggested),
      initialDirectory: p.dirname(suggested),
      acceptedTypeGroups: const [
        XTypeGroup(label: 'MP4', extensions: ['mp4']),
      ],
    );
    if (loc == null) return;
    var out = loc.path;
    if (p.extension(out).isEmpty) out = '$out.mp4';
    if (p.equals(out, c.videoPath!)) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Choose a different output file')),
        );
      }
      return;
    }
    await c.upscaleVideo(out);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (!c.caps.supportsVideo) {
      return const DropZone(
        icon: Icons.desktop_windows_outlined,
        title: 'Video upscaling needs a desktop build',
        subtitle:
            'It uses ffmpeg to decode and encode frames, which is only '
            'wired up on Windows, macOS and Linux.',
      );
    }
    if (!c.ffmpegChecked) {
      return const Center(child: CircularProgressIndicator());
    }
    if (c.ffmpeg == null) return _FfmpegMissing(onRetry: c.detectFfmpeg);

    final busy = c.isBusy;
    final videoBusy = c.busy == UpscaleBusy.video;
    final info = c.videoInfo;
    final decode = c.videoDecodeSize;
    final frames = c.videoFramesToProcess;
    final tilesPerFrame = decode == null
        ? null
        : c.tiler.tilesFor(decode.$1, decode.$2);

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
              icon: const Icon(Icons.movie_outlined, size: 18),
              label: const Text('Open video'),
            ),
            FilledButton.icon(
              onPressed: busy || info == null || !c.modelReady ? null : _start,
              icon: const Icon(Icons.auto_awesome, size: 18),
              label: const Text('Upscale ×4…'),
            ),
            if (videoBusy)
              TextButton.icon(
                onPressed: c.cancel,
                icon: const Icon(Icons.stop_circle_outlined, size: 18),
                label: const Text('Cancel'),
              ),
            if (c.videoOutputPath != null && !busy) ...[
              OutlinedButton.icon(
                onPressed: () => openWithSystem(c.videoOutputPath!),
                icon: const Icon(Icons.play_circle_outline, size: 18),
                label: const Text('Play result'),
              ),
              IconButton(
                tooltip: 'Show in folder',
                onPressed: () => revealInFileManager(c.videoOutputPath!),
                icon: const Icon(Icons.folder_open_outlined),
              ),
            ],
          ],
        ),
        const SizedBox(height: 8),
        if (info != null)
          Wrap(
            spacing: 16,
            runSpacing: 4,
            children: [
              _kv(
                theme,
                'Source',
                '${info.width}×${info.height} '
                    '${info.codec} @ ${info.fps.toStringAsFixed(2)} fps',
              ),
              _kv(theme, 'Frames', info.estimatedFrames?.toString() ?? '?'),
              _kv(
                theme,
                'Audio',
                info.hasAudio ? 'yes (re-encoded AAC)' : 'no',
              ),
              if (decode != null)
                _kv(
                  theme,
                  'Pipeline',
                  '${decode.$1}×${decode.$2} → '
                      '${decode.$1 * c.tiler.scale}×${decode.$2 * c.tiler.scale}',
                ),
              if (tilesPerFrame != null)
                _kv(
                  theme,
                  'Work',
                  '$tilesPerFrame tiles/frame × ${frames ?? '?'} frames',
                ),
            ],
          ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 16,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            _Dropdown<double>(
              label: 'Input scale',
              value: c.videoScale,
              items: {1.0: '100%', 0.75: '75%', 0.5: '50%', 0.25: '25%'},
              onChanged: busy ? null : (v) => c.setParams(videoScale: v),
              help:
                  'Downscale frames before ×4 upscaling. 50% gives a 2× '
                  'final size and 4× fewer tiles.',
            ),
            _Dropdown<int>(
              label: 'Frames',
              value: c.maxFrames,
              items: const {
                0: 'All',
                30: 'First 30',
                60: 'First 60',
                150: 'First 150',
                300: 'First 300',
              },
              onChanged: busy ? null : (v) => c.setParams(maxFrames: v),
            ),
            Text(
              c.ffmpeg!.version,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
        const SizedBox(height: 8),
        StatusLine(controller: c),
        if (videoBusy && c.videoProgress != null && frames != null)
          _Eta(controller: c, totalFrames: frames),
        const SizedBox(height: 8),
        Expanded(
          child: DropTarget(
            onDragEntered: (_) => setState(() => _dragging = true),
            onDragExited: (_) => setState(() => _dragging = false),
            onDragDone: (d) {
              setState(() => _dragging = false);
              if (d.files.isNotEmpty && !busy) c.setVideo(d.files.first.path);
            },
            child: c.videoInputFrame != null && c.videoOutputFrame != null
                ? CompareView(
                    before: c.videoInputFrame!,
                    after: c.videoOutputFrame!,
                    beforeLabel: 'Frame in',
                    afterLabel: 'Frame ×4',
                  )
                : DropZone(
                    highlight: _dragging,
                    icon: Icons.video_file_outlined,
                    title: c.videoPath == null
                        ? 'Drop a video here'
                        : p.basename(c.videoPath!),
                    subtitle: c.videoPath == null
                        ? 'Frames stream through ffmpeg → LiteRT → ffmpeg '
                              '(H.264 + original audio), with no temp files.'
                        : 'A live before/after preview appears here while '
                              'frames are processed.',
                  ),
          ),
        ),
      ],
    );
  }

  Widget _kv(ThemeData theme, String k, String v) => Text.rich(
    TextSpan(
      children: [
        TextSpan(
          text: '$k  ',
          style: TextStyle(color: theme.colorScheme.onSurfaceVariant),
        ),
        TextSpan(text: v),
      ],
    ),
    style: theme.textTheme.bodySmall,
  );
}

class _Eta extends StatelessWidget {
  const _Eta({required this.controller, required this.totalFrames});
  final UpscaleController controller;
  final int totalFrames;

  @override
  Widget build(BuildContext context) {
    final m = controller.videoProgress!;
    if (m.frame == 0) return const SizedBox.shrink();
    final perFrame = m.elapsedMs / m.frame;
    final remaining = (totalFrames - m.frame) * perFrame;
    final fps = 1000 / perFrame;
    return Text(
      '${fps.toStringAsFixed(2)} fps · ~${(remaining / 1000).ceil()} s left',
      style: Theme.of(context).textTheme.bodySmall,
    );
  }
}

class _Dropdown<T> extends StatelessWidget {
  const _Dropdown({
    required this.label,
    required this.value,
    required this.items,
    required this.onChanged,
    this.help,
  });

  final String label;
  final T value;
  final Map<T, String> items;
  final ValueChanged<T>? onChanged;
  final String? help;

  @override
  Widget build(BuildContext context) {
    final row = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('$label: ', style: Theme.of(context).textTheme.labelMedium),
        DropdownButton<T>(
          value: items.containsKey(value) ? value : items.keys.first,
          isDense: true,
          underline: const SizedBox.shrink(),
          items: [
            for (final e in items.entries)
              DropdownMenuItem(value: e.key, child: Text(e.value)),
          ],
          onChanged: onChanged == null ? null : (v) => onChanged!(v as T),
        ),
      ],
    );
    return help == null ? row : Tooltip(message: help!, child: row);
  }
}

class _FfmpegMissing extends StatelessWidget {
  const _FfmpegMissing({required this.onRetry});
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final install = Platform.isWindows
        ? 'winget install Gyan.FFmpeg'
        : Platform.isMacOS
        ? 'brew install ffmpeg'
        : 'sudo apt install ffmpeg';
    return Stack(
      children: [
        DropZone(
          icon: Icons.extension_off_outlined,
          title: 'ffmpeg not found',
          subtitle:
              'Video upscaling uses the system ffmpeg/ffprobe.\n'
              'Install it with `$install`, or set its path in Settings, '
              'then re-check.',
        ),
        Positioned(
          bottom: 24,
          left: 0,
          right: 0,
          child: Center(
            child: FilledButton.tonalIcon(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh, size: 18),
              label: const Text('Re-check'),
            ),
          ),
        ),
      ],
    );
  }
}
