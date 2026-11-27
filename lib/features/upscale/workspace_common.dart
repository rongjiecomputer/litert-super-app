import 'package:material_ui/material_ui.dart';

import '../../core/metrics/metrics_panel.dart';
import 'upscale_controller.dart';

/// Progress bar plus status / error text for the running job.
class StatusLine extends StatelessWidget {
  const StatusLine({super.key, required this.controller});
  final UpscaleController controller;

  @override
  Widget build(BuildContext context) {
    final c = controller;
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: 4,
          child: c.isBusy
              ? LinearProgressIndicator(
                  value: c.progress,
                  borderRadius: BorderRadius.circular(2),
                )
              : null,
        ),
        const SizedBox(height: 6),
        if (c.error != null)
          SelectableText(
            c.error!,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.error,
            ),
            maxLines: 4,
          )
        else
          Text(
            c.status ?? ' ',
            style: theme.textTheme.bodySmall,
            overflow: TextOverflow.ellipsis,
          ),
        if (!c.isBusy && c.lastRun != null) ...[
          const SizedBox(height: 6),
          CostStrip(run: c.lastRun!),
        ],
      ],
    );
  }
}

class PreviewFrame extends StatelessWidget {
  const PreviewFrame({super.key, required this.child, this.highlight = false});
  final Widget child;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: highlight ? cs.primary : Colors.transparent,
          width: 2,
        ),
      ),
      child: ClipRRect(borderRadius: BorderRadius.circular(10), child: child),
    );
  }
}

class DropZone extends StatelessWidget {
  const DropZone({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.highlight = false,
  });

  final IconData icon;
  final String title, subtitle;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: highlight
            ? cs.primaryContainer.withValues(alpha: 0.4)
            : cs.surfaceContainerLow,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: highlight ? cs.primary : cs.outlineVariant,
          width: highlight ? 2 : 1,
        ),
      ),
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 48, color: cs.primary),
              const SizedBox(height: 12),
              Text(title, style: theme.textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(
                subtitle,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: cs.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
