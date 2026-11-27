import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Filesystem layout shared with the `litert-lm` CLI.
///
/// Desktop: `~/.litert-lm/{models,downloading,cache}`.
/// Mobile:  `<app support dir>/litert-lm/...` (no shared home directory).
///
/// The app only writes inside `models/<id>/`, `downloading/` and `cache/`;
/// it never touches the CLI's `config.json` or `history`.
class AppPaths {
  AppPaths(this.root);

  final String root;

  String get models => p.join(root, 'models');
  String get downloading => p.join(root, 'downloading');
  String get cache => p.join(root, 'cache');

  String modelDir(String modelId) => p.join(models, modelId);

  static Future<AppPaths> resolve() async {
    final isMobile = Platform.isAndroid || Platform.isIOS;
    final mobileBase = isMobile
        ? (await getApplicationSupportDirectory()).path
        : null;
    final root = resolveRoot(
      isMobile: isMobile,
      isWindows: Platform.isWindows,
      environment: Platform.environment,
      mobileBase: mobileBase,
    );
    final paths = AppPaths(root);
    for (final dir in [paths.models, paths.downloading, paths.cache]) {
      Directory(dir).createSync(recursive: true);
    }
    return paths;
  }

  /// Pure resolver, separated for unit tests.
  static String resolveRoot({
    required bool isMobile,
    required bool isWindows,
    required Map<String, String> environment,
    String? mobileBase,
  }) {
    if (isMobile) {
      if (mobileBase == null) {
        throw ArgumentError('mobileBase is required on mobile platforms');
      }
      return p.join(mobileBase, 'litert-lm');
    }
    final home = isWindows
        ? (environment['USERPROFILE'] ??
              _join(environment['HOMEDRIVE'], environment['HOMEPATH']) ??
              environment['HOME'])
        : environment['HOME'];
    if (home == null || home.isEmpty) {
      throw StateError('Cannot determine the user home directory');
    }
    final ctx = isWindows ? p.windows : p.posix;
    return ctx.join(home, '.litert-lm');
  }

  static String? _join(String? a, String? b) =>
      a == null || b == null ? null : '$a$b';
}
