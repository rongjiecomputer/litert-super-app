import 'dart:io';

import 'package:path/path.dart' as p;

bool get isDesktop =>
    Platform.isWindows || Platform.isMacOS || Platform.isLinux;

/// Opens a folder (or a file's folder) in the OS file manager.
Future<void> revealInFileManager(String path) async {
  if (Platform.isWindows) {
    if (FileSystemEntity.isFileSync(path)) {
      await Process.run('explorer', ['/select,', path]);
    } else {
      await Process.run('explorer', [path]);
    }
  } else if (Platform.isMacOS) {
    await Process.run(
      'open',
      FileSystemEntity.isFileSync(path) ? ['-R', path] : [path],
    );
  } else if (Platform.isLinux) {
    final dir = FileSystemEntity.isDirectorySync(path) ? path : p.dirname(path);
    await Process.run('xdg-open', [dir]);
  }
}

/// Opens a file with its default application.
Future<void> openWithSystem(String path) async {
  if (Platform.isWindows) {
    await Process.run('cmd', ['/c', 'start', '', path]);
  } else if (Platform.isMacOS) {
    await Process.run('open', [path]);
  } else if (Platform.isLinux) {
    await Process.run('xdg-open', [path]);
  }
}
