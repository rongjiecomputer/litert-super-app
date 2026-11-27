import 'package:material_ui/material_ui.dart';

import 'package:path/path.dart' as p;

import '../features/home/home_page.dart';
import '../features/models/models_page.dart';
import '../features/settings/settings_page.dart';
import 'services.dart';
import 'use_cases.dart';

class _Dest {
  const _Dest(this.label, this.icon, this.selectedIcon, this.builder);
  final String label;
  final IconData icon, selectedIcon;
  final WidgetBuilder builder;
}

/// Adaptive scaffold: left NavigationRail on desktop / wide windows,
/// bottom NavigationBar on phones.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;
  final _built = <int>{0};

  late final List<_Dest> _dests = [
    _Dest(
      'Home',
      Icons.home_outlined,
      Icons.home,
      (_) => HomePage(onOpen: _openUseCase),
    ),
    for (final u in useCases.where((u) => u.available))
      _Dest(u.title, u.icon, u.selectedIcon, u.builder!),
    _Dest(
      'Models',
      Icons.inventory_2_outlined,
      Icons.inventory_2,
      (_) => const ModelsPage(),
    ),
    _Dest(
      'Settings',
      Icons.settings_outlined,
      Icons.settings,
      (_) => const SettingsPage(),
    ),
  ];

  @override
  void initState() {
    super.initState();
    // Jump straight to the use case that can open a command-line file.
    final file = AppServices.instance.launchFile;
    if (file != null) {
      final ext = p.extension(file).toLowerCase().replaceFirst('.', '');
      final u = useCases.where((u) => u.available && u.opens.contains(ext));
      if (u.isNotEmpty) _openUseCase(u.first);
    }
  }

  void _openUseCase(UseCase u) {
    final i = _dests.indexWhere((d) => d.label == u.title);
    if (i >= 0) _select(i);
  }

  void _select(int i) => setState(() {
    _index = i;
    _built.add(i);
  });

  @override
  Widget build(BuildContext context) {
    // Pages are built lazily on first visit, then kept alive so running
    // jobs and loaded models survive navigation.
    final body = IndexedStack(
      index: _index,
      children: [
        for (var i = 0; i < _dests.length; i++)
          _built.contains(i)
              ? _dests[i].builder(context)
              : const SizedBox.shrink(),
      ],
    );

    return LayoutBuilder(
      builder: (context, box) {
        if (box.maxWidth < 720) {
          return Scaffold(
            body: SafeArea(child: body),
            bottomNavigationBar: NavigationBar(
              selectedIndex: _index,
              onDestinationSelected: _select,
              destinations: [
                for (final d in _dests)
                  NavigationDestination(
                    icon: Icon(d.icon),
                    selectedIcon: Icon(d.selectedIcon),
                    label: d.label,
                  ),
              ],
            ),
          );
        }
        final extended = box.maxWidth >= 1280;
        final theme = Theme.of(context);
        return Scaffold(
          body: Row(
            children: [
              NavigationRail(
                extended: extended,
                minExtendedWidth: 200,
                selectedIndex: _index,
                onDestinationSelected: _select,
                labelType: extended
                    ? NavigationRailLabelType.none
                    : NavigationRailLabelType.all,
                backgroundColor: theme.colorScheme.surfaceContainerLow,
                leading: Padding(
                  padding: const EdgeInsets.fromLTRB(8, 12, 8, 20),
                  child: _Brand(extended: extended),
                ),
                destinations: [
                  for (final d in _dests)
                    NavigationRailDestination(
                      icon: Icon(d.icon),
                      selectedIcon: Icon(d.selectedIcon),
                      label: Text(d.label),
                    ),
                ],
              ),
              const VerticalDivider(width: 1, thickness: 1),
              Expanded(child: body),
            ],
          ),
        );
      },
    );
  }
}

class _Brand extends StatelessWidget {
  const _Brand({required this.extended});
  final bool extended;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final logo = Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [theme.colorScheme.primary, theme.colorScheme.tertiary],
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Icon(Icons.bolt, color: theme.colorScheme.onPrimary),
    );
    if (!extended) return logo;
    return SizedBox(
      width: 176,
      child: Row(
        children: [
          logo,
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('LiteRT', style: theme.textTheme.titleMedium),
              Text('On-device AI', style: theme.textTheme.bodySmall),
            ],
          ),
        ],
      ),
    );
  }
}
