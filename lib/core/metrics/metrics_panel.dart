import 'package:material_ui/material_ui.dart';

import '../widgets/section_card.dart';
import 'run_metrics.dart';

/// Shows model load cost, the latest run's cost breakdown and run history.
class MetricsPanel extends StatelessWidget {
  const MetricsPanel({
    super.key,
    required this.loadInfo,
    required this.lastRun,
    required this.history,
    this.unitLabel = 'tile',
    this.runSection,
  });

  final LoadInfo? loadInfo;
  final RunMetrics? lastRun;
  final List<RunMetrics> history;
  final String unitLabel;

  /// Replaces the image-oriented "last run" and history cards (e.g. audio
  /// use cases with their own metrics).
  final Widget? runSection;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionCard(
          title: 'Model',
          icon: Icons.memory,
          child: loadInfo == null
              ? const _Empty('Load the model to see compile cost.')
              : _LoadSection(loadInfo!),
        ),
        const SizedBox(height: 12),
        if (runSection != null)
          runSection!
        else
        SectionCard(
          title: 'Inference cost (last run)',
          icon: Icons.speed,
          child: lastRun == null
              ? const _Empty('Run a task to see timings and memory.')
              : _RunSection(lastRun!, unitLabel),
        ),
        if (runSection == null && history.isNotEmpty) ...[
          const SizedBox(height: 12),
          SectionCard(
            title: 'History',
            icon: Icons.history,
            child: _History(history),
          ),
        ],
      ],
    );
  }
}

class _Empty extends StatelessWidget {
  const _Empty(this.text);
  final String text;
  @override
  Widget build(BuildContext context) => Text(
    text,
    style: Theme.of(context).textTheme.bodySmall
        ?.copyWith(color: Theme.of(context).colorScheme.onSurfaceVariant),
  );
}

class _LoadSection extends StatelessWidget {
  const _LoadSection(this.info);
  final LoadInfo info;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(info.runtimeLabel, style: Theme.of(context).textTheme.bodyMedium),
        const SizedBox(height: 8),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final a in info.actual)
              Chip(
                visualDensity: VisualDensity.compact,
                avatar: const Icon(Icons.bolt, size: 16),
                label: Text(a),
              ),
            if (info.didFallback)
              Tooltip(
                message: info.fallbackReason ?? 'Fell back to CPU',
                child: Chip(
                  visualDensity: VisualDensity.compact,
                  backgroundColor: cs.errorContainer,
                  avatar: Icon(Icons.warning_amber, size: 16, color: cs.error),
                  label: Text(
                    'Fallback from ${info.requested.join('+')}',
                    style: TextStyle(color: cs.onErrorContainer),
                  ),
                ),
              ),
            if (info.fullyAccelerated == true)
              const Chip(
                visualDensity: VisualDensity.compact,
                avatar: Icon(Icons.verified_outlined, size: 16),
                label: Text('Fully accelerated'),
              ),
          ],
        ),
        const SizedBox(height: 8),
        _Grid([
          ('Load / compile', formatMs(info.loadMs)),
          ('Warm-up run', formatMs(info.warmupMs)),
          ('RAM at load', formatBytes(info.rssDeltaBytes)),
        ]),
      ],
    );
  }
}

class _RunSection extends StatelessWidget {
  const _RunSection(this.m, this.unit);
  final RunMetrics m;
  final String unit;

  @override
  Widget build(BuildContext context) {
    final t = m.tileStats;
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(m.task, style: theme.textTheme.bodyMedium),
        Text(
          '${m.optionsSummary}  →  ${m.actualAccelerators}',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 10),
        _Hero(
          total: formatMs(m.totalMs),
          secondary: m.fps != null
              ? '${m.fps!.toStringAsFixed(2)} fps'
              : '${m.megapixelsOut.toStringAsFixed(1)} MP out',
        ),
        const SizedBox(height: 10),
        _Grid([
          if (m.loadMs != null) ('Model load', formatMs(m.loadMs!)),
          ('Inference (sum)', formatMs(t.sum)),
          ('${unit[0].toUpperCase()}${unit.substring(1)}s', '${m.tiles}'),
          if (m.frames != null) ('Frames', '${m.frames}'),
          ('Per $unit avg', formatMs(t.mean)),
          ('Per $unit p50', formatMs(t.p50)),
          ('Per $unit p90', formatMs(t.p90)),
          ('Min / max', '${formatMs(t.min)} / ${formatMs(t.max)}'),
          ('Throughput', '${m.tilesPerSecond.toStringAsFixed(1)} $unit/s'),
          (m.frames != null ? 'ffmpeg I/O' : 'Decode', formatMs(m.decodeMs)),
          ('Pre-process', formatMs(m.preprocessMs)),
          (
            m.frames != null ? 'Post + encode' : 'Post-process',
            formatMs(m.postprocessMs),
          ),
        ]),
        const Divider(height: 24),
        _Grid([
          ('RAM before', formatBytes(m.rssBefore)),
          ('RAM peak', formatBytes(m.rssPeak)),
          ('Peak Δ', '+${formatBytes(m.rssDelta)}'),
          ('RAM after', formatBytes(m.rssAfter)),
          ('Process max RSS', formatBytes(m.maxRss)),
        ]),
      ],
    );
  }
}

/// One-line cost summary for the latest run, shown next to the result.
class CostStrip extends StatelessWidget {
  const CostStrip({super.key, required this.run, this.unitLabel = 'tile'});
  final RunMetrics run;
  final String unitLabel;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    Widget chip(IconData icon, String text, String tooltip) => Tooltip(
      message: tooltip,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: cs.secondaryContainer,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: cs.onSecondaryContainer),
            const SizedBox(width: 6),
            Text(
              text,
              style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: cs.onSecondaryContainer,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: [
        chip(
          Icons.timer_outlined,
          formatMs(run.totalMs),
          'Total wall time'
          '${run.loadMs == null ? '' : ' (incl. ${formatMs(run.loadMs!)} model load)'}',
        ),
        chip(
          Icons.speed,
          '${formatMs(run.tileStats.p50)}/$unitLabel',
          'Median inference latency per $unitLabel (p90 ${formatMs(run.tileStats.p90)})',
        ),
        if (run.fps != null)
          chip(
            Icons.movie_outlined,
            '${run.fps!.toStringAsFixed(2)} fps',
            'End-to-end frames per second',
          ),
        chip(
          Icons.memory,
          formatBytes(run.rssPeak),
          'Peak process RAM (+${formatBytes(run.rssDelta)} during the run)',
        ),
        chip(Icons.bolt, run.actualAccelerators, run.optionsSummary),
      ],
    );
  }
}

class _Hero extends StatelessWidget {
  const _Hero({required this.total, required this.secondary});
  final String total, secondary;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.baseline,
      textBaseline: TextBaseline.alphabetic,
      children: [
        Text(
          total,
          style: theme.textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.w600,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
        const SizedBox(width: 8),
        Text('total', style: theme.textTheme.bodySmall),
        const Spacer(),
        Text(secondary, style: theme.textTheme.titleSmall),
      ],
    );
  }
}

class _Grid extends StatelessWidget {
  const _Grid(this.items);
  final List<(String, String)> items;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return LayoutBuilder(
      builder: (context, c) {
        final cols = c.maxWidth > 420 ? 3 : 2;
        final w = (c.maxWidth - (cols - 1) * 8) / cols;
        return Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final (label, value) in items)
              SizedBox(
                width: w,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: theme.textTheme.labelSmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                    Text(
                      value,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontFeatures: const [FontFeature.tabularFigures()],
                      ),
                    ),
                  ],
                ),
              ),
          ],
        );
      },
    );
  }
}

class _History extends StatelessWidget {
  const _History(this.runs);
  final List<RunMetrics> runs;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final small = theme.textTheme.bodySmall;
    final muted = small?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    return Column(
      children: [
        for (final r in runs.take(12))
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        r.task,
                        style: small,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Text(
                        r.optionsSummary,
                        style: muted,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(formatMs(r.totalMs), style: small),
                    Text(
                      'p50 ${formatMs(r.tileStats.p50)} · '
                      '${formatBytes(r.rssPeak)}',
                      style: muted,
                    ),
                  ],
                ),
              ],
            ),
          ),
      ],
    );
  }
}
