import 'package:material_ui/material_ui.dart';

import '../metrics/run_metrics.dart';
import '../models/model_manager.dart';
import '../models/model_spec.dart';

/// Inline prompt to download a missing model (with progress).
class ModelBanner extends StatelessWidget {
  const ModelBanner({super.key, required this.manager, required this.spec});

  final ModelManager manager;
  final ModelSpec spec;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: manager,
      builder: (context, _) {
        final state = manager.stateOf(spec);
        if (state.status == ModelStatus.ready) return const SizedBox.shrink();
        final cs = Theme.of(context).colorScheme;
        final downloading = state.status == ModelStatus.downloading;
        return Card(
          elevation: 0,
          color: cs.secondaryContainer,
          margin: const EdgeInsets.only(bottom: 12),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Icon(Icons.download_for_offline_outlined, color: cs.secondary),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '${spec.name} is not downloaded '
                        '(~${formatBytes(spec.approxBytes)})',
                        style: Theme.of(context).textTheme.titleSmall,
                      ),
                      Text(
                        state.error ?? manager.pathOf(spec),
                        style: Theme.of(context).textTheme.bodySmall,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (downloading) ...[
                        const SizedBox(height: 6),
                        LinearProgressIndicator(value: state.progress),
                      ],
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                downloading
                    ? TextButton(
                        onPressed: () => manager.cancelDownload(spec),
                        child: const Text('Cancel'),
                      )
                    : FilledButton.icon(
                        onPressed: () => manager.download(spec),
                        icon: const Icon(Icons.download, size: 18),
                        label: const Text('Download'),
                      ),
              ],
            ),
          ),
        );
      },
    );
  }
}
