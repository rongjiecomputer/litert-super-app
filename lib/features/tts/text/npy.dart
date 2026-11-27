import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Minimal NumPy `.npy` / stored (uncompressed) `.npz` reader.
///
/// Supports little-endian `<f4` and `<f2` (up-cast to float32), C order.
class NpyArray {
  NpyArray(this.shape, this.data);

  final List<int> shape;
  final Float32List data;

  int get rowLength => shape.length <= 1 ? 1 : shape.skip(1).fold(1, (a, b) => a * b);

  /// View of row [i] along the first axis (no copy).
  Float32List row(int i) =>
      Float32List.sublistView(data, i * rowLength, (i + 1) * rowLength);

  static NpyArray readFile(String path) => parse(File(path).readAsBytesSync());

  static NpyArray parse(Uint8List bytes) {
    final h = NpyHeader.parse(bytes);
    final count = h.shape.fold(1, (a, b) => a * b);
    final body = ByteData.sublistView(bytes, h.dataOffset);
    return NpyArray(h.shape, _decode(body, h.dtype, count));
  }

  /// Reads all arrays of a stored `.npz` archive.
  static Map<String, NpyArray> readNpz(String path) {
    final bytes = File(path).readAsBytesSync();
    final out = <String, NpyArray>{};
    for (final e in _zipEntries(bytes)) {
      final name = e.name.endsWith('.npy')
          ? e.name.substring(0, e.name.length - 4)
          : e.name;
      out[name] = parse(Uint8List.sublistView(bytes, e.offset, e.offset + e.size));
    }
    return out;
  }
}

/// Parsed `.npy` header.
class NpyHeader {
  NpyHeader(this.dtype, this.shape, this.dataOffset);

  final String dtype;
  final List<int> shape;
  final int dataOffset;

  int get itemSize => switch (dtype) {
    '<f4' => 4,
    '<f2' => 2,
    _ => throw FormatException('Unsupported npy dtype $dtype'),
  };

  static NpyHeader parse(Uint8List bytes) {
    if (bytes.length < 10 ||
        bytes[0] != 0x93 ||
        ascii.decode(bytes.sublist(1, 6)) != 'NUMPY') {
      throw const FormatException('Not an .npy file');
    }
    final major = bytes[6];
    final bd = ByteData.sublistView(bytes);
    final int headerLen, start;
    if (major == 1) {
      headerLen = bd.getUint16(8, Endian.little);
      start = 10;
    } else {
      headerLen = bd.getUint32(8, Endian.little);
      start = 12;
    }
    final header = latin1.decode(bytes.sublist(start, start + headerLen));
    final descr = RegExp(r"'descr':\s*'([^']+)'").firstMatch(header)?.group(1);
    final fortran = RegExp(r"'fortran_order':\s*(True|False)")
        .firstMatch(header)
        ?.group(1);
    final shapeStr = RegExp(r"'shape':\s*\(([^)]*)\)").firstMatch(header)?.group(1);
    if (descr == null || shapeStr == null) {
      throw FormatException('Bad npy header: $header');
    }
    if (fortran == 'True') throw const FormatException('Fortran order npy');
    final shape = shapeStr
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .map(int.parse)
        .toList();
    return NpyHeader(descr, shape, start + headerLen);
  }
}

/// Random-access row reader for large `.npy` tables that should not be
/// loaded into memory (e.g. the 622 MB Qwen text-embedding table).
class NpyRowReader {
  NpyRowReader._(this._file, this.header);

  final RandomAccessFile _file;
  final NpyHeader header;

  int get rows => header.shape.first;
  int get rowLength => header.shape.skip(1).fold(1, (a, b) => a * b);

  static NpyRowReader open(String path) {
    final f = File(path).openSync();
    final head = f.readSync(4096);
    return NpyRowReader._(f, NpyHeader.parse(head));
  }

  /// Reads row [i] as float32.
  Float32List row(int i) {
    if (i < 0 || i >= rows) throw RangeError.range(i, 0, rows - 1);
    final bytesPerRow = rowLength * header.itemSize;
    _file.setPositionSync(header.dataOffset + i * bytesPerRow);
    final raw = _file.readSync(bytesPerRow);
    return _decode(ByteData.sublistView(raw), header.dtype, rowLength);
  }

  void close() => _file.closeSync();
}

Float32List _decode(ByteData body, String dtype, int count) {
  final aligned = body.offsetInBytes % 4 == 0;
  if (aligned && Endian.host == Endian.little) {
    // Fast paths: typed views over the raw bytes (npy pads headers to 64).
    switch (dtype) {
      case '<f4':
        return Float32List.fromList(
          body.buffer.asFloat32List(body.offsetInBytes, count),
        );
      case '<f2':
        final src = body.buffer.asUint16List(body.offsetInBytes, count);
        final out = Float32List(count);
        final t = _f16Table;
        for (var i = 0; i < count; i++) {
          out[i] = t[src[i]];
        }
        return out;
    }
  }
  switch (dtype) {
    case '<f4':
      final out = Float32List(count);
      for (var i = 0; i < count; i++) {
        out[i] = body.getFloat32(i * 4, Endian.little);
      }
      return out;
    case '<f2':
      final out = Float32List(count);
      for (var i = 0; i < count; i++) {
        out[i] = halfToFloat(body.getUint16(i * 2, Endian.little));
      }
      return out;
    default:
      throw FormatException('Unsupported npy dtype $dtype');
  }
}

final _f16Table = _buildF16Table();

Float32List _buildF16Table() {
  final t = Float32List(65536);
  for (var h = 0; h < 65536; h++) {
    final s = (h >> 15) & 1;
    final e = (h >> 10) & 0x1f;
    final m = h & 0x3ff;
    double v;
    if (e == 0) {
      v = m / 1024.0 * 6.103515625e-05; // subnormal: m * 2^-24
    } else if (e == 31) {
      v = m == 0 ? double.infinity : double.nan;
    } else {
      v = (1 + m / 1024.0) * _pow2(e - 15);
    }
    t[h] = s == 1 ? -v : v;
  }
  return t;
}

double _pow2(int e) {
  var r = 1.0;
  if (e >= 0) {
    for (var i = 0; i < e; i++) {
      r *= 2;
    }
  } else {
    for (var i = 0; i < -e; i++) {
      r /= 2;
    }
  }
  return r;
}

/// IEEE 754 half → double.
double halfToFloat(int h) => _f16Table[h & 0xffff];

class _ZipEntry {
  _ZipEntry(this.name, this.offset, this.size);
  final String name;
  final int offset;
  final int size;
}

/// Lists stored (method 0) entries of a zip via its central directory.
/// Handles ZIP64 sizes, which NumPy writes for large members.
List<_ZipEntry> _zipEntries(Uint8List bytes) {
  final bd = ByteData.sublistView(bytes);
  var eocd = -1;
  for (var i = bytes.length - 22; i >= 0 && i >= bytes.length - 65557; i--) {
    if (bd.getUint32(i, Endian.little) == 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw const FormatException('Not a zip archive');
  final count = bd.getUint16(eocd + 10, Endian.little);
  var p = bd.getUint32(eocd + 16, Endian.little);
  final out = <_ZipEntry>[];
  for (var n = 0; n < count; n++) {
    if (bd.getUint32(p, Endian.little) != 0x02014b50) {
      throw const FormatException('Bad zip central directory');
    }
    final method = bd.getUint16(p + 10, Endian.little);
    var compSize = bd.getUint32(p + 20, Endian.little);
    final nameLen = bd.getUint16(p + 28, Endian.little);
    final extraLen = bd.getUint16(p + 30, Endian.little);
    final commentLen = bd.getUint16(p + 32, Endian.little);
    var localOffset = bd.getUint32(p + 42, Endian.little);
    final name = utf8.decode(bytes.sublist(p + 46, p + 46 + nameLen));
    // ZIP64 extra field.
    var e = p + 46 + nameLen;
    final extraEnd = e + extraLen;
    while (e + 4 <= extraEnd) {
      final id = bd.getUint16(e, Endian.little);
      final size = bd.getUint16(e + 2, Endian.little);
      if (id == 0x0001) {
        var q = e + 4;
        final uncompSize32 = bd.getUint32(p + 24, Endian.little);
        if (uncompSize32 == 0xffffffff) q += 8;
        if (compSize == 0xffffffff) {
          compSize = bd.getUint64(q, Endian.little);
          q += 8;
        }
        if (localOffset == 0xffffffff) {
          localOffset = bd.getUint64(q, Endian.little);
        }
      }
      e += 4 + size;
    }
    if (method != 0) {
      throw FormatException('Compressed npz member $name is not supported');
    }
    final lNameLen = bd.getUint16(localOffset + 26, Endian.little);
    final lExtraLen = bd.getUint16(localOffset + 28, Endian.little);
    out.add(_ZipEntry(name, localOffset + 30 + lNameLen + lExtraLen, compSize));
    p += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

/// Encodes a little-endian float32 C-order `.npy` (format 1.0).
Uint8List encodeNpyF32(Float32List data, List<int> shape) {
  final shapeStr = shape.length == 1 ? '(${shape[0]},)' : '(${shape.join(', ')})';
  var header = "{'descr': '<f4', 'fortran_order': False, 'shape': $shapeStr, }";
  // Pad so the data starts on a 64-byte boundary (header ends with \n).
  final unpadded = 10 + header.length + 1;
  header = header.padRight(header.length + (64 - unpadded % 64) % 64);
  header += '\n';
  final out = BytesBuilder()
    ..add([0x93, ...ascii.encode('NUMPY'), 1, 0])
    ..add([header.length & 0xff, header.length >> 8])
    ..add(latin1.encode(header))
    ..add(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
  return out.toBytes();
}
