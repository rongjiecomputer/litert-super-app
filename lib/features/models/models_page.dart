import 'package:material_ui/material_ui.dart';

import '../../app/services.dart';
import '../../core/metrics/run_metrics.dart';
import '../../core/models/model_catalog.dart';
import '../../core/models/model_manager.dart';
import '../../core/models/model_spec.dart';
import '../../core/system.dart';
import '../../core/widgets/page_header.dart';

class ModelsPage extends StatelessWidget {
  const ModelsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final services = AppServices.instance;
    final m = services.models;
    final theme = Theme.of(context);
    return ListenableBuilder(
      listenable: m,
      builder: (context, _) => ListView(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
        children: [
          PageHeader(
            icon: Icons.inventory_2,
            title: 'Models',
            subtitle: services.paths.models,
            actions: [
              IconButton(
                tooltip: 'Rescan',
                onPressed: m.refresh,
                icon: const Icon(Icons.refresh),
              ),
              if (isDesktop)
                IconButton(
                  tooltip: 'Open folder',
                  onPressed: () => revealInFileManager(services.paths.models),
                  icon: const Icon(Icons.folder_open_outlined),
                ),
            ],
          ),
          Text('Used by this app', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          for (final spec in ModelCatalog.all) _ModelTile(m, spec),
          if (m.foreign.isNotEmpty) ...[
            const SizedBox(height: 24),
            Text(
              'Other models in this folder',
              style: theme.textTheme.titleMedium,
            ),
            Text(
              'Shared with the litert-lm CLI and other tools; listed for '
              'reference only.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            for (final f in m.foreign)
              Card(
                elevation: 0,
                color: theme.colorScheme.surfaceContainerLow,
                child: ListTile(
                  leading: const Icon(Icons.description_outlined),
                  title: Text(f.id),
                  subtitle: Text(f.files.join(', ')),
                  trailing: Text(formatBytes(f.bytes)),
                ),
              ),
          ],
        ],
      ),
    );
  }
}

class _ModelTile extends StatelessWidget {
  const _ModelTile(this.manager, this.spec);
  final ModelManager manager;
  final ModelSpec spec;

  @override
  Widget build(BuildContext context) {
    final state = manager.stateOf(spec);
    final theme = Theme.of(context);
    final path = manager.pathOf(spec);
    final (icon, label) = switch (state.status) {
      ModelStatus.ready => (
        Icons.check_circle,
        'Ready · ${formatBytes(state.sizeOnDisk ?? 0)}',
      ),
      ModelStatus.downloading => (
        Icons.downloading,
        'Downloading ${formatBytes(state.received)}'
            '${state.total == null ? '' : ' / ${formatBytes(state.total!)}'}',
      ),
      ModelStatus.missing => (
        Icons.cloud_download_outlined,
        'Not downloaded · ~${formatBytes(manager.missingBytes(spec))}',
      ),
      ModelStatus.error => (Icons.error_outline, 'Error: ${state.error}'),
    };
    return Card(
      elevation: 0,
      color: theme.colorScheme.surfaceContainerLow,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: theme.colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(spec.name, style: theme.textTheme.titleMedium),
                ),
                Icon(icon, size: 18, color: theme.colorScheme.primary),
                const SizedBox(width: 6),
                Flexible(child: Text(label, overflow: TextOverflow.ellipsis)),
              ],
            ),
            const SizedBox(height: 6),
            Text(spec.description, style: theme.textTheme.bodySmall),
            const SizedBox(height: 6),
            SelectableText(
              '${spec.hfRepo} · '
              '${spec.extraFiles.isEmpty ? spec.fileName : '${spec.files.length} files'}'
              '${spec.license == null ? '' : ' · ${spec.license}'}\n$path',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            if (state.status == ModelStatus.downloading) ...[
              const SizedBox(height: 8),
              LinearProgressIndicator(value: state.progress),
            ],
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              children: [
                switch (state.status) {
                  ModelStatus.downloading => OutlinedButton(
                    onPressed: () => manager.cancelDownload(spec),
                    child: const Text('Cancel'),
                  ),
                  ModelStatus.ready => OutlinedButton.icon(
                    onPressed: () => _confirmDelete(context),
                    icon: const Icon(Icons.delete_outline, size: 18),
                    label: const Text('Delete'),
                  ),
                  _ => FilledButton.icon(
                    onPressed: () => manager.download(spec),
                    icon: const Icon(Icons.download, size: 18),
                    label: const Text('Download'),
                  ),
                },
                if (state.status == ModelStatus.ready && isDesktop)
                  TextButton.icon(
                    onPressed: () => revealInFileManager(path),
                    icon: const Icon(Icons.folder_open_outlined, size: 18),
                    label: const Text('Show in folder'),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _confirmDelete(BuildContext context) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete ${spec.name}?'),
        content: Text(
          'This removes ${spec.extraFiles.isEmpty ? manager.pathOf(spec) : 'its ${spec.files.length} files from ${manager.paths.modelDir(spec.id)}'}. '
          'You can download it again '
          'later.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok == true) await manager.delete(spec);
  }
}
