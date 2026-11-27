import 'dart:io';

import 'package:shared_preferences/shared_preferences.dart';

import '../core/models/model_manager.dart';
import '../core/paths.dart';
import '../core/settings.dart';

/// App-wide singletons, created once in `main()`.
class AppServices {
  AppServices._(this.paths, this.settings, this.models);

  static late final AppServices instance;

  final AppPaths paths;
  final AppSettings settings;
  final ModelManager models;

  /// File passed on the command line ("Open with…"); consumed by the
  /// first use case that accepts it.
  String? launchFile;

  static Future<void> init({List<String> launchArgs = const []}) async {
    final paths = await AppPaths.resolve();
    final prefs = await SharedPreferences.getInstance();
    instance = AppServices._(paths, AppSettings(prefs), ModelManager(paths))
      ..launchFile = launchArgs.where((a) => File(a).existsSync()).firstOrNull;
  }
}
