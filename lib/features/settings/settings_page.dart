import 'package:material_ui/material_ui.dart';

import '../../app/services.dart';
import '../../core/system.dart';
import '../../core/widgets/page_header.dart';
import '../upscale/ffmpeg.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  final settings = AppServices.instance.settings;
  late final _ffmpegCtrl = TextEditingController(text: settings.ffmpegPath);
  String? _ffmpegStatus;
  bool _checking = false;

  @override
  void initState() {
    super.initState();
    if (isDesktop) _check();
  }

  @override
  void dispose() {
    _ffmpegCtrl.dispose();
    super.dispose();
  }

  Future<void> _check() async {
    setState(() => _checking = true);
    settings.ffmpegPath = _ffmpegCtrl.text;
    final tools = await FfmpegTools.locate(override: settings.ffmpegPath);
    if (!mounted) return;
    setState(() {
      _checking = false;
      _ffmpegStatus = tools == null
          ? 'Not found'
          : '${tools.version}\n${tools.ffmpeg}';
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final paths = AppServices.instance.paths;
    return ListenableBuilder(
      listenable: settings,
      builder: (context, _) => ListView(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
        children: [
          const PageHeader(icon: Icons.settings, title: 'Settings'),
          Text('Appearance', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          SegmentedButton<ThemeMode>(
            segments: const [
              ButtonSegment(
                value: ThemeMode.system,
                icon: Icon(Icons.brightness_auto),
                label: Text('System'),
              ),
              ButtonSegment(
                value: ThemeMode.light,
                icon: Icon(Icons.light_mode),
                label: Text('Light'),
              ),
              ButtonSegment(
                value: ThemeMode.dark,
                icon: Icon(Icons.dark_mode),
                label: Text('Dark'),
              ),
            ],
            selected: {settings.themeMode},
            onSelectionChanged: (s) => settings.themeMode = s.first,
          ),
          const SizedBox(height: 28),
          Text('Storage', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.folder_outlined),
            title: const Text('Model root'),
            subtitle: SelectableText(paths.root),
            trailing: isDesktop
                ? IconButton(
                    tooltip: 'Open',
                    onPressed: () => revealInFileManager(paths.root),
                    icon: const Icon(Icons.open_in_new),
                  )
                : null,
          ),
          if (isDesktop) ...[
            const SizedBox(height: 20),
            Text('Video (ffmpeg)', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _ffmpegCtrl,
                    decoration: const InputDecoration(
                      labelText: 'ffmpeg path (optional)',
                      hintText: 'Leave empty to search PATH',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    onSubmitted: (_) => _check(),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton.tonal(
                  onPressed: _checking ? null : _check,
                  child: const Text('Detect'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              _checking ? 'Checking…' : (_ffmpegStatus ?? ''),
              style: theme.textTheme.bodySmall,
            ),
          ],
        ],
      ),
    );
  }
}
