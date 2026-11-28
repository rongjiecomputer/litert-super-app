import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

import '../paths.dart';
import 'litertlm.dart';
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

  /// Path of the model's primary file.
  String pathOf(ModelSpec spec) =>
      p.join(paths.modelDir(spec.id), spec.fileName);

  /// Path of any file of the model (`relPath` may contain `/`).
  String fileOf(ModelSpec spec, String relPath) =>
      p.joinAll([paths.modelDir(spec.id), ...relPath.split('/')]);

  bool hasFile(ModelSpec spec, String relPath) =>
      File(fileOf(spec, relPath)).existsSync();

  bool isReady(ModelSpec spec) => stateOf(spec).status == ModelStatus.ready;

  /// Approximate bytes still to download for [spec]'s required files.
  int missingBytes(ModelSpec spec) {
    var bytes = 0;
    var bundleMissing = false;
    for (final f in spec.requiredFiles) {
      if (hasFile(spec, f.relPath)) continue;
      if (spec.bundleEntry(f.relPath) != null) {
        bundleMissing = true;
      } else {
        bytes += f.approxBytes;
      }
    }
    return bytes + (bundleMissing ? spec.bundle!.approxBytes : 0);
  }

  void refresh() {
    for (final spec in ModelCatalog.all) {
      if (_downloads.containsKey(spec.id)) continue;
      var ready = true;
      var bytes = 0;
      for (final f in spec.files) {
        final file = File(fileOf(spec, f.relPath));
        if (file.existsSync()) {
          bytes += file.lengthSync();
        } else if (!f.optional) {
          ready = false;
        }
      }
      _states[spec.id] = ready
          ? ModelState(status: ModelStatus.ready, sizeOnDisk: bytes)
          : ModelState(status: ModelStatus.missing, sizeOnDisk: bytes);
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

  /// Downloads every missing required file of [spec], one after another.
  /// Files that come from a `.litertlm` bundle are unpacked from a single
  /// download of the bundle, which is then discarded.
  Future<void> download(ModelSpec spec) async {
    if (_downloads.containsKey(spec.id)) return;
    final dl = _Download();
    _downloads[spec.id] = dl;
    final jobs = <_Fetch>[];
    for (final f in spec.requiredFiles) {
      if (hasFile(spec, f.relPath) || spec.bundleEntry(f.relPath) != null) continue;
      jobs.add(
        _Fetch(f.uri, f.relPath.replaceAll('/', '_'), f.approxBytes, (part) async {
          final target = File(fileOf(spec, f.relPath));
          target.parent.createSync(recursive: true);
          if (target.existsSync()) target.deleteSync();
          // Same volume (both under the root) so rename is atomic.
          part.renameSync(target.path);
        }),
      );
    }
    final bundle = spec.bundle;
    final entries = [
      if (bundle != null)
        for (final e in bundle.entries)
          if (!hasFile(spec, e.relPath)) e,
    ];
    if (bundle != null && entries.isNotEmpty) {
      jobs.add(
        _Fetch(bundle.uri, 'bundle.litertlm', bundle.approxBytes, (part) async {
          final jobsToRun = [
            for (final e in entries)
              (fileOf(spec, e.relPath), e.dataType, e.key, e.value),
          ];
          await _unpackInIsolate(part.path, jobsToRun);
          part.deleteSync();
        }),
      );
    }
    var total = jobs.fold<int>(0, (a, j) => a + j.approxBytes);
    var done = 0;
    _set(
      spec,
      ModelState(status: ModelStatus.downloading, total: total),
    );
    String? partPath;
    IOSink? sink;
    try {
      for (final job in jobs) {
        partPath = p.join(
          paths.downloading,
          '${spec.id}--${job.partName}.part',
        );
        final res = await dl.client.send(http.Request('GET', job.uri));
        if (res.statusCode != 200) {
          throw HttpException('HTTP ${res.statusCode}', uri: job.uri);
        }
        // Correct the estimate once the real size is known.
        if (res.contentLength != null) {
          total += res.contentLength! - job.approxBytes;
        }
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
                received: done + received,
                total: total,
              ),
            );
          }
        }
        await sink.close();
        sink = null;
        done += received;
        await job.finish(part);
        partPath = null;
      }
      _downloads.remove(spec.id);
      refresh();
    } catch (e) {
      await sink?.close();
      if (partPath != null) {
        final part = File(partPath);
        if (part.existsSync()) part.deleteSync();
      }
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

  /// Copies a user-provided file (e.g. a converted add-on graph) into the
  /// model folder as [relPath].
  Future<void> importFile(ModelSpec spec, String relPath, String source) async {
    final target = File(fileOf(spec, relPath));
    target.parent.createSync(recursive: true);
    await File(source).copy(target.path);
    refresh();
  }

  void cancelDownload(ModelSpec spec) {
    final dl = _downloads[spec.id];
    if (dl == null) return;
    dl.cancelled = true;
    dl.client.close();
  }

  /// Deletes the model's files (only the ones the catalog knows about) and
  /// any folders left empty.
  Future<void> delete(ModelSpec spec) async {
    final root = Directory(paths.modelDir(spec.id));
    for (final f in spec.files) {
      final file = File(fileOf(spec, f.relPath));
      if (file.existsSync()) file.deleteSync();
    }
    if (root.existsSync()) {
      final dirs = root.listSync(recursive: true).whereType<Directory>().toList()
        ..sort((a, b) => b.path.length.compareTo(a.path.length));
      for (final d in [...dirs, root]) {
        if (d.existsSync() && d.listSync().isEmpty) d.deleteSync();
      }
    }
    refresh();
  }

  void _set(ModelSpec spec, ModelState state) {
    _states[spec.id] = state;
    notifyListeners();
  }
}

/// Unpacks a bundle off the UI isolate. A top-level function so the isolate
/// closure captures only its (sendable) arguments, not the download state.
Future<void> _unpackInIsolate(
  String bundlePath,
  List<(String target, int dataType, String key, String value)> jobs,
) => Isolate.run(() => unpackLitertLm(bundlePath, jobs));

/// One HTTP download: stored in `downloading/<model>--<partName>.part`, then
/// handed to [finish] to move or unpack.
class _Fetch {
  _Fetch(this.uri, this.partName, this.approxBytes, this.finish);

  final Uri uri;
  final String partName;
  final int approxBytes;
  final Future<void> Function(File part) finish;
}

class _Download {
  final client = http.Client();
  bool cancelled = false;
}

class _Cancelled implements Exception {
  const _Cancelled();
}
