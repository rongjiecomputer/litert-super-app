import 'package:material_ui/material_ui.dart';

import 'app/services.dart';
import 'app/shell.dart';

Future<void> main([List<String> args = const []]) async {
  WidgetsFlutterBinding.ensureInitialized();
  await AppServices.init(launchArgs: args);
  runApp(const LiteRtApp());
}

class LiteRtApp extends StatelessWidget {
  const LiteRtApp({super.key});

  static const _seed = Color(0xFF1A73E8);

  @override
  Widget build(BuildContext context) {
    final settings = AppServices.instance.settings;
    return ListenableBuilder(
      listenable: settings,
      builder: (context, _) => MaterialApp(
        title: 'LiteRT Super App',
        debugShowCheckedModeBanner: false,
        themeMode: settings.themeMode,
        theme: ThemeData(colorSchemeSeed: _seed, useMaterial3: true),
        darkTheme: ThemeData(
          colorSchemeSeed: _seed,
          brightness: Brightness.dark,
          useMaterial3: true,
        ),
        home: const AppShell(),
      ),
    );
  }
}
