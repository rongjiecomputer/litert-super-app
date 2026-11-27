// Entrypoint with Flutter Driver enabled, used for automated UI checks:
//   flutter run -d windows -t test_driver/app_main.dart
import 'package:ai_litert/main.dart' as app;
import 'package:flutter_driver/driver_extension.dart';

Future<void> main(List<String> args) async {
  enableFlutterDriverExtension();
  await app.main(args);
}
