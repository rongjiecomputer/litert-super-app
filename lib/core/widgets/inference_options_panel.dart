import 'package:material_ui/material_ui.dart';

import '../inference/inference_options.dart';
import '../metrics/run_metrics.dart';
import 'section_card.dart';

/// Reusable per-use-case inference configuration panel.
class InferenceOptionsPanel extends StatelessWidget {
  const InferenceOptionsPanel({
    super.key,
    required this.options,
    required this.caps,
    required this.onChanged,
    required this.enabled,
    required this.loadInfo,
    required this.needsReload,
    required this.onLoad,
    required this.onUnload,
    this.extra = const [],
  });

  final InferenceOptions options;
  final PlatformCaps caps;
  final ValueChanged<InferenceOptions> onChanged;
  final bool enabled;
  final LoadInfo? loadInfo;
  final bool needsReload;
  final VoidCallback? onLoad;
  final VoidCallback? onUnload;

  /// Use-case specific controls appended below the runtime options.
  final List<Widget> extra;

  @override
  Widget build(BuildContext context) {
    final o = options;
    final accels = caps.accelerators(o.runtime);
    final isCompiled = o.runtime == RuntimeKind.compiledModel;
    final theme = Theme.of(context);

    Widget seg<T extends Object>(
      List<T> values,
      T selected,
      String Function(T) label,
      ValueChanged<T> onSelect, {
      bool Function(T)? isEnabled,
    }) {
      return SizedBox(
        width: double.infinity,
        child: SegmentedButton<T>(
          showSelectedIcon: false,
          style: const ButtonStyle(visualDensity: VisualDensity.compact),
          segments: [
            for (final v in values)
              ButtonSegment(
                value: v,
                label: Text(label(v)),
                enabled: isEnabled?.call(v) ?? true,
              ),
          ],
          selected: {selected},
          onSelectionChanged: enabled ? (s) => onSelect(s.first) : null,
        ),
      );
    }

    final loaded = loadInfo;
    return SectionCard(
      title: 'Inference options',
      icon: Icons.tune,
      trailing: loaded == null
          ? const _Chip('Not loaded', Icons.radio_button_unchecked)
          : needsReload
          ? const _Chip('Reload needed', Icons.sync_problem, warn: true)
          : const _Chip('Loaded', Icons.check_circle_outline),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          LabeledField(
            label: 'Runtime',
            help:
                'CompiledModel: LiteRT 2.x API with CPU/GPU/NPU accelerators.\n'
                'Interpreter: classic TFLite API with XNNPACK (CPU) or Metal '
                'delegate.',
            child: seg(
              RuntimeKind.values,
              o.runtime,
              (v) => v.label,
              (v) => onChanged(o.copyWith(runtime: v).sanitized(caps)),
            ),
          ),
          LabeledField(
            label: 'Accelerator',
            help: switch (caps.os) {
              'windows' => 'Windows GPU runs on WebGPU over Direct3D 12.',
              'macos' => 'macOS GPU runs on Metal; NPU uses Core ML.',
              'android' =>
                'Android GPU uses OpenCL/GL; NPU needs a vendor '
                    'runtime.',
              _ => null,
            },
            child: seg(
              AcceleratorKind.values,
              o.accelerator,
              (v) => v.label,
              (v) => onChanged(o.copyWith(accelerator: v).sanitized(caps)),
              isEnabled: accels.contains,
            ),
          ),
          LabeledField(
            label: 'Precision',
            help:
                'FP16 halves GPU arithmetic precision (faster, may shift '
                'colours). On CPU it sets XNNPACK force-FP16.',
            child: seg(
              PrecisionKind.values,
              o.precision,
              (v) => v.label,
              (v) => onChanged(o.copyWith(precision: v)),
              isEnabled: (v) =>
                  v == PrecisionKind.fp32 || caps.supportsPrecision(o),
            ),
          ),
          if (isCompiled) ...[
            LabeledField(
              label: 'Tensor buffers',
              help:
                  'Managed: LiteRT-owned buffers (copy via lock/unlock).\n'
                  'Host memory: app-owned aligned buffers, which can cut copies '
                  'on some GPUs.',
              child: seg(
                BufferModeKind.values,
                o.bufferMode,
                (v) => v.label,
                (v) => onChanged(o.copyWith(bufferMode: v)),
              ),
            ),
            if (o.accelerator != AcceleratorKind.cpu)
              SwitchListTile(
                dense: true,
                contentPadding: EdgeInsets.zero,
                title: const Text('Fall back to CPU if compilation fails'),
                value: o.cpuFallback,
                onChanged: enabled
                    ? (v) => onChanged(o.copyWith(cpuFallback: v))
                    : null,
              ),
          ],
          if (!isCompiled && o.accelerator == AcceleratorKind.cpu)
            LabeledField(
              label: 'CPU threads: ${o.cpuThreads}',
              child: Slider(
                value: o.cpuThreads.toDouble(),
                min: 1,
                max: caps.maxThreads.toDouble(),
                divisions: caps.maxThreads > 1 ? caps.maxThreads - 1 : null,
                label: '${o.cpuThreads}',
                onChanged: enabled
                    ? (v) => onChanged(o.copyWith(cpuThreads: v.round()))
                    : null,
              ),
            ),
          ...extra,
          const SizedBox(height: 4),
          Row(
            children: [
              Expanded(
                child: FilledButton.tonalIcon(
                  onPressed: enabled ? onLoad : null,
                  icon: const Icon(Icons.memory, size: 18),
                  label: Text(loaded == null ? 'Load model' : 'Reload model'),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                tooltip: 'Unload model',
                onPressed: enabled && loaded != null ? onUnload : null,
                icon: const Icon(Icons.eject_outlined),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Changing options reloads the model on the next run.',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip(this.label, this.icon, {this.warn = false});
  final String label;
  final IconData icon;
  final bool warn;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final color = warn ? cs.tertiary : cs.onSurfaceVariant;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color),
        ),
      ],
    );
  }
}
