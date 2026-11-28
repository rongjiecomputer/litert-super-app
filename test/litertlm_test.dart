import 'dart:io';
import 'dart:typed_data';

import 'package:ai_litert/core/models/litertlm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

// test/fixtures/tiny.litertlm was made with `litert-lm pack` from two
// GenericBinaryData sections: name=alpha ("hello-section-one") and
// name=beta, kind=table (bytes 0..255 repeated 20 times).
const fixture = 'test/fixtures/tiny.litertlm';

void main() {
  late Directory tmp;
  setUp(() => tmp = Directory.systemTemp.createTempSync('litertlm_test'));
  tearDown(() => tmp.deleteSync(recursive: true));

  test('lists sections with their string metadata', () {
    final f = LitertLmFile.open(fixture);
    final named = f.sections.where((s) => s.metadata.containsKey('name')).toList();
    expect(named.map((s) => s.metadata['name']), ['alpha', 'beta']);
    expect(named.every((s) => s.dataType == LitertLmType.genericBinaryData), isTrue);
    expect(named[0].length, 17);
    expect(named[1].length, 5120);
    expect(named[1].metadata['kind'], 'table');
  });

  test('find selects by type, key and value', () {
    final f = LitertLmFile.open(fixture);
    expect(f.find(LitertLmType.genericBinaryData, 'name', 'beta')!.length, 5120);
    expect(f.find(LitertLmType.genericBinaryData, 'name', 'gamma'), isNull);
    expect(f.find(LitertLmType.tfliteModel, 'name', 'alpha'), isNull);
  });

  test('extract copies the payload bytes and leaves no temp file', () {
    final f = LitertLmFile.open(fixture);
    final alpha = p.join(tmp.path, 'sub', 'alpha.bin');
    final beta = p.join(tmp.path, 'beta.bin');
    f.extract(f.find(LitertLmType.genericBinaryData, 'name', 'alpha')!, alpha);
    f.extract(f.find(LitertLmType.genericBinaryData, 'name', 'beta')!, beta);
    expect(File(alpha).readAsStringSync(), 'hello-section-one');
    expect(
      File(beta).readAsBytesSync(),
      Uint8List.fromList([for (var r = 0; r < 20; r++) ...List.generate(256, (i) => i)]),
    );
    expect(File('$alpha.tmp').existsSync(), isFalse);
  });

  test('unpackLitertLm extracts jobs and reports a missing section', () {
    final out = p.join(tmp.path, 'a.bin');
    unpackLitertLm(fixture, [(out, LitertLmType.genericBinaryData, 'name', 'alpha')]);
    expect(File(out).readAsStringSync(), 'hello-section-one');
    expect(
      () => unpackLitertLm(fixture, [
        (p.join(tmp.path, 'x.bin'), LitertLmType.genericBinaryData, 'name', 'nope'),
      ]),
      throwsFormatException,
    );
  });

  test('rejects files that are not .litertlm bundles', () {
    final bad = File(p.join(tmp.path, 'bad.litertlm'))
      ..writeAsBytesSync(Uint8List(64)..setAll(0, 'NOTLITERT'.codeUnits));
    expect(() => LitertLmFile.open(bad.path), throwsFormatException);
  });
}
