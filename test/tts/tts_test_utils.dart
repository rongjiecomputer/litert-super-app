import 'dart:io';

import 'package:path/path.dart' as p;

/// `~/.litert-lm/models/<id>/<file>` if it exists (tests that need real
/// model files skip otherwise).
String? modelFile(String modelId, String relPath) {
  final home = Platform.environment['USERPROFILE'] ??
      Platform.environment['HOME'] ??
      '';
  final path = p.joinAll([home, '.litert-lm', 'models', modelId, ...relPath.split('/')]);
  return File(path).existsSync() ? path : null;
}

String golden(String name) => p.join('test', 'goldens', 'tts', name);
