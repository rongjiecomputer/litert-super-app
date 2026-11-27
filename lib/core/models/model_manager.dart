import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

import '../paths.dart';
import 'model_catalog.dart';
import 'model_spec.dart';

enum ModelStatus { missing, downloading, ready, error }

class ModelState {
  const ModelState({
    required this.status,
    this.received = 0,
    this.total,
    this.sizeOnDisk,
    this.error,
  });

  final ModelStatus status;
  final int received;
  final int? total;
  final int? sizeOnDisk;
  final String? error;

  double? get progress =>
      total == null || total == 0 ? null : (received / total!).clamp(0, 1);
}

/// A model folder found under `models/` that is not in the app catalog
/// (e.g. LiteRT-LM `.litertlm` bundles downloaded by the CLI).
class ForeignModel {
  const ForeignModel(this.id, this.files, this.bytes);
  final String id;
  final List<String> files;
  final int bytes;
}

/// Tracks, downloads and deletes catalog models under [AppPaths.models].
class ModelManager extends ChangeNotifier {
  ModelManager(this.paths) {
    refresh();
  }

  final AppPaths paths;
  final Map<String, ModelState> _states = {};
  final Map<String, _Download> _downloads = {};
  List<ForeignModel> foreign = const [];

  ModelState stateOf(ModelSpec spec) =>
      _states[spec.id] ?? const ModelState(status: ModelStatus.missing);

  String pathOf(ModelSpec spec) =>
      p.join(paths.modelDir(spec.id), spec.fileName);

  bool isReady(ModelSpec spec) => stateOf(spec).status == ModelStatus.ready;

  void refresh() {
    for (final spec in ModelCatalog.all) {
      if (_downloads.containsKey(spec.id)) continue;
      final f = File(pathOf(spec));
      _states[spec.id] = f.existsSync()
          ? ModelState(status: ModelStatus.ready, sizeOnDisk: f.lengthSync())
          : const ModelState(status: ModelStatus.missing);
    }
    foreign = _scanForeign();
    notifyListeners();
  }

  List<ForeignModel> _scanForeign() {
    final known = ModelCatalog.all.map((m) => m.id).toSet();
    final dir = Directory(paths.models);
    if (!dir.existsSync()) return const [];
    final out = <ForeignModel>[];
    for (final e in dir.listSync().whereType<Directory>()) {
      final id = p.basename(e.path);
      if (known.contains(id)) continue;
      var bytes = 0;
      final files = <String>[];
      for (final f in e.listSync(recursive: true).whereType<File>()) {
        files.add(p.relative(f.path, from: e.path));
        bytes += f.lengthSync();
      }
      out.add(ForeignModel(id, files, bytes));
    }
    out.sort((a, b) => a.id.compareTo(b.id));
    return out;
  }

  Future<void> download(ModelSpec spec) async {
    if (_downloads.containsKey(spec.id)) return;
    final dl = _Download();
    _downloads[spec.id] = dl;
    final partPath = p.join(
      paths.downloading,
      '${spec.id}--${spec.fileName}.part',
    );
    _set(spec, const ModelState(status: ModelStatus.downloading));
    IOSink? sink;
    try {
      final res = await dl.client.send(http.Request('GET', spec.downloadUrl));
      if (res.statusCode != 200) {
        throw HttpException('HTTP ${res.statusCode}', uri: spec.downloadUrl);
      }
      final total = res.contentLength ?? spec.approxBytes;
      Directory(paths.downloading).createSync(recursive: true);
      final part = File(partPath);
      sink = part.openWrite();
      var received = 0;
      var lastNotify = DateTime.now();
      await for (final chunk in res.stream) {
        if (dl.cancelled) throw const _Cancelled();
        sink.add(chunk);
        received += chunk.length;
        final now = DateTime.now();
        if (now.difference(lastNotify).inMilliseconds > 100) {
          lastNotify = now;
          _set(
            spec,
            ModelState(
              status: ModelStatus.downloading,
              received: received,
              total: total,
            ),
          );
        }
      }
      await sink.close();
      sink = null;
      final target = File(pathOf(spec));
      target.parent.createSync(recursive: true);
      if (target.existsSync()) target.deleteSync();
      // Same volume (both under the root) so rename is atomic.
      part.renameSync(target.path);
      _downloads.remove(spec.id);
      refresh();
    } catch (e) {
      await sink?.close();
      final part = File(partPath);
      if (part.existsSync()) part.deleteSync();
      _downloads.remove(spec.id);
      if (e is _Cancelled) {
        refresh();
      } else {
        _set(spec, ModelState(status: ModelStatus.error, error: '$e'));
      }
    } finally {
      dl.client.close();
    }
  }

  void cancelDownload(ModelSpec spec) {
    final dl = _downloads[spec.id];
    if (dl == null) return;
    dl.cancelled = true;
    dl.client.close();
  }

  Future<void> delete(ModelSpec spec) async {
    final dir = Directory(paths.modelDir(spec.id));
    final f = File(pathOf(spec));
    if (f.existsSync()) f.deleteSync();
    if (dir.existsSync() && dir.listSync().isEmpty) dir.deleteSync();
    refresh();
  }

  void _set(ModelSpec spec, ModelState state) {
    _states[spec.id] = state;
    notifyListeners();
  }
}

class _Download {
  final client = http.Client();
  bool cancelled = false;
}

class _Cancelled implements Exception {
  const _Cancelled();
}
