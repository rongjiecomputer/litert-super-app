import 'dart:io';

import 'package:material_ui/material_ui.dart';

import '../../app/services.dart';
import '../../app/use_cases.dart';
import '../../core/inference/inference_options.dart';
import '../../core/litert/ffi.dart';
import '../../core/models/model_manager.dart';
import '../../core/widgets/page_header.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key, required this.onOpen});
  final ValueChanged<UseCase> onOpen;

  @override
  Widget build(BuildContext context) {
    final services = AppServices.instance;
    final caps = PlatformCaps.current();
    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
      children: [
        const PageHeader(
          icon: Icons.bolt,
          title: 'LiteRT Super App',
          subtitle:
              'A showcase of on-device AI use cases. Nothing leaves this '
              'machine; models live in a shared local folder.',
        ),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            _Fact(
              Icons.computer,
              '${Platform.operatingSystem} '
              '· ${Platform.numberOfProcessors} CPU threads',
            ),
            _Fact(
              Icons.bolt,
              'LiteRT CompiledModel: '
              '${caps.accelerators.map((a) => a.label).join(' / ')}',
            ),
            _Fact(Icons.memory, 'LiteRT ${_version()}'),
            _Fact(Icons.folder_outlined, services.paths.models),
          ],
        ),
        const SizedBox(height: 24),
        Text('Use cases', style: theme.textTheme.titleMedium),
        const SizedBox(height: 12),
        ListenableBuilder(
          listenable: services.models,
          builder: (context, _) => LayoutBuilder(
            builder: (context, box) {
              final cols = (box.maxWidth / 320).floor().clamp(1, 4);
              final w = (box.maxWidth - (cols - 1) * 16) / cols;
              return Wrap(
                spacing: 16,
                runSpacing: 16,
                children: [
                  for (final u in useCases)
                    SizedBox(
                      width: w,
                      child: _UseCaseCard(
                        useCase: u,
                        models: services.models,
                        onOpen: () => onOpen(u),
                      ),
                    ),
                ],
              );
            },
          ),
        ),
      ],
    );
  }

  static String _version() {
    try {
      return LiteRtApi.instance.abiVersion;
    } catch (_) {
      return 'n/a';
    }
  }
}

class _Fact extends StatelessWidget {
  const _Fact(this.icon, this.text);
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) => Chip(
    avatar: Icon(icon, size: 16),
    label: Text(text),
    visualDensity: VisualDensity.compact,
  );
}

class _UseCaseCard extends StatelessWidget {
  const _UseCaseCard({
    required this.useCase,
    required this.models,
    required this.onOpen,
  });

  final UseCase useCase;
  final ModelManager models;
  final VoidCallback onOpen;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final u = useCase;
    final ready = u.models.every(models.isReady);
    return Opacity(
      opacity: u.available ? 1 : 0.6,
      child: Card(
        elevation: 0,
        margin: EdgeInsets.zero,
        color: cs.surfaceContainerLow,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: cs.outlineVariant),
        ),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: u.available ? onOpen : null,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: SizedBox(
              height: 150,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(u.selectedIcon, color: cs.primary),
                      const Spacer(),
                      if (!u.available)
                        const Chip(
                          label: Text('Planned'),
                          visualDensity: VisualDensity.compact,
                        )
                      else
                        Chip(
                          visualDensity: VisualDensity.compact,
                          avatar: Icon(
                            ready ? Icons.check_circle : Icons.download,
                            size: 16,
                          ),
                          label: Text(ready ? 'Ready' : 'Needs model'),
                        ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(u.title, style: theme.textTheme.titleMedium),
                  const SizedBox(height: 4),
                  Expanded(
                    child: Text(
                      u.summary,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: cs.onSurfaceVariant,
                      ),
                      overflow: TextOverflow.fade,
                    ),
                  ),
                  Wrap(
                    spacing: 6,
                    children: [
                      for (final t in u.tags)
                        Text(
                          '#$t',
                          style: theme.textTheme.labelSmall?.copyWith(
                            color: cs.primary,
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
