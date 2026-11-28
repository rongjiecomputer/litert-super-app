import 'dart:io';

import 'package:ai_litert/core/models/litertlm.dart';
import 'package:ai_litert/core/models/model_catalog.dart';
import 'package:ai_litert/core/models/model_manager.dart';
import 'package:ai_litert/core/models/model_spec.dart';
import 'package:ai_litert/core/paths.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

void main() {
  late Directory tmp;
  late ModelManager manager;
  setUp(() {
    tmp = Directory.systemTemp.createTempSync('model_manager_test');
    final paths = AppPaths(tmp.path);
    for (final d in [paths.models, paths.downloading, paths.cache]) {
      Directory(d).createSync(recursive: true);
    }
    manager = ModelManager(paths);
  });
  tearDown(() {
    manager.dispose();
    tmp.deleteSync(recursive: true);
  });

  group('bundle specs', () {
    final kokoro = ModelCatalog.kokoro82m;

    test('the Kokoro spec lists the bundle entries, not the bundle itself', () {
      final names = kokoro.files.map((f) => f.relPath).toList();
      expect(names, containsAll(['acoustic.tflite', 'vocoder.tflite', 'voices/af_heart.bin']));
      expect(names, containsAll(['dp_g2p_litert.tflite', 'config.json', 'cmudict.txt']));
      expect(names.where((n) => n.endsWith('.litertlm')), isEmpty);
      expect(kokoro.bundleEntry('vocoder.tflite'), isNotNull);
      expect(kokoro.bundleEntry('config.json'), isNull);
    });

    test('missingBytes counts the bundle once plus the plain files', () {
      final bundle = kokoro.bundle!;
      final extras = kokoro.extraFiles.fold<int>(0, (a, f) => a + f.approxBytes);
      expect(manager.missingBytes(kokoro), bundle.approxBytes + extras);
      expect(kokoro.approxBytes, bundle.approxBytes + extras);

      // With every bundle entry on disk only the plain files remain.
      for (final e in bundle.entries) {
        final f = File(manager.fileOf(kokoro, e.relPath))..createSync(recursive: true);
        f.writeAsBytesSync([0]);
      }
      expect(manager.missingBytes(kokoro), extras);
    });
  });

  test('download fetches the bundle once and unpacks the entries', () async {
    final bytes = File('test/fixtures/tiny.litertlm').readAsBytesSync();
    var requests = 0;
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    addTearDown(() => server.close(force: true));
    server.listen((req) {
      requests++;
      req.response
        ..contentLength = bytes.length
        ..add(bytes);
      req.response.close();
    });
    final spec = ModelSpec(
      id: 'fixture',
      name: 'Fixture',
      description: '',
      hfRepo: 'none/none',
      fileName: 'alpha.bin',
      approxBytes: bytes.length,
      bundle: ModelBundle('http://localhost:${server.port}/tiny.litertlm', bytes.length, const [
        BundleEntry('alpha.bin', 17,
            dataType: LitertLmType.genericBinaryData, key: 'name', value: 'alpha'),
        BundleEntry('tables/beta.bin', 5120,
            dataType: LitertLmType.genericBinaryData, key: 'name', value: 'beta'),
      ]),
    );

    await manager.download(spec);

    expect(requests, 1);
    expect(manager.stateOf(spec).status, isNot(ModelStatus.error), reason: manager.stateOf(spec).error);
    expect(File(manager.fileOf(spec, 'alpha.bin')).readAsStringSync(), 'hello-section-one');
    expect(File(manager.fileOf(spec, 'tables/beta.bin')).lengthSync(), 5120);
    // The bundle itself is not kept, and no partial download is left.
    expect(Directory(manager.paths.downloading).listSync(), isEmpty);
    expect(Directory(manager.paths.modelDir('fixture')).listSync(recursive: true).whereType<File>().map((f) => p.basename(f.path)).toSet(), {'alpha.bin', 'beta.bin'});
  });
}
