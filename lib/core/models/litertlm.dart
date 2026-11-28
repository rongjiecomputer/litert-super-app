import 'dart:io';
import 'dart:typed_data';

/// Section payload types of a `.litertlm` container (`AnySectionDataType` in
/// LiteRT-LM's `litertlm_header_schema.fbs`); only the ones we select on.
abstract final class LitertLmType {
  static const genericBinaryData = 1;
  static const tfliteModel = 3;
}

/// One section of a `.litertlm` bundle.
class LitertLmSection {
  const LitertLmSection({
    required this.begin,
    required this.end,
    required this.dataType,
    required this.metadata,
  });

  /// Byte range of the payload in the file.
  final int begin, end;

  /// A [LitertLmType] value.
  final int dataType;

  /// String entries of the section's key/value metadata (`model_type`,
  /// `name`, ...).
  final Map<String, String> metadata;

  int get length => end - begin;
}

/// Reader for LiteRT-LM's `.litertlm` container, enough to do what
/// `litert-lm unpack` does: list the sections and copy their payloads out.
///
/// Layout: `"LITERTLM"`, u32 major/minor/patch, 4 bytes padding, u64 header
/// end offset, then a FlatBuffer (`LiteRTLMMetaData`) up to that offset. The
/// payloads follow at the offsets the header lists. Sections of an
/// `.litertlm` TTS bundle are plain `.tflite` files, voice tables and
/// front-end data.
class LitertLmFile {
  LitertLmFile._(this.path, this.sections);

  final String path;
  final List<LitertLmSection> sections;

  static const _magic = 'LITERTLM';
  static const _supportedMajor = 1;

  static LitertLmFile open(String path) {
    final raf = File(path).openSync();
    try {
      final head = raf.readSync(32);
      if (head.length < 32 || String.fromCharCodes(head.sublist(0, 8)) != _magic) {
        throw FormatException('Not a .litertlm file', path);
      }
      final bd = ByteData.sublistView(head);
      final major = bd.getUint32(8, Endian.little);
      if (major != _supportedMajor) {
        throw FormatException('Unsupported .litertlm version $major', path);
      }
      final headerEnd = bd.getUint64(24, Endian.little);
      if (headerEnd < 32 || headerEnd > raf.lengthSync()) {
        throw FormatException('Invalid .litertlm header size', path);
      }
      final header = raf.readSync(headerEnd - 32);
      return LitertLmFile._(path, _parseHeader(header));
    } finally {
      raf.closeSync();
    }
  }

  /// The first section of [dataType] whose metadata contains [key] = [value].
  LitertLmSection? find(int dataType, String key, String value) {
    for (final s in sections) {
      if (s.dataType == dataType && s.metadata[key] == value) return s;
    }
    return null;
  }

  /// Copies [section]'s payload to [target], creating parent folders. The
  /// file is written under `<target>.tmp` and renamed, so a partial copy is
  /// never left at [target].
  void extract(LitertLmSection section, String target) {
    final out = File(target);
    out.parent.createSync(recursive: true);
    final tmp = File('$target.tmp');
    final src = File(path).openSync();
    final dst = tmp.openSync(mode: FileMode.write);
    try {
      src.setPositionSync(section.begin);
      var left = section.length;
      while (left > 0) {
        final chunk = src.readSync(left < (1 << 20) ? left : (1 << 20));
        if (chunk.isEmpty) throw FileSystemException('Truncated .litertlm', path);
        dst.writeFromSync(chunk);
        left -= chunk.length;
      }
    } catch (_) {
      dst.closeSync();
      src.closeSync();
      if (tmp.existsSync()) tmp.deleteSync();
      rethrow;
    }
    dst.closeSync();
    src.closeSync();
    if (out.existsSync()) out.deleteSync();
    tmp.renameSync(target);
  }

  static List<LitertLmSection> _parseHeader(Uint8List bytes) {
    final root = _Table.root(bytes);
    final meta = root.table(1); // section_metadata
    if (meta == null) return const [];
    final out = <LitertLmSection>[];
    for (final obj in meta.tables(0)) {
      final items = <String, String>{};
      for (final kv in obj.tables(0)) {
        final key = kv.string(0);
        // VData union: type byte at field 1, value table at field 2. Only
        // StringValue (type 9) entries are kept.
        if (key != null && kv.u8(1) == 9) {
          final v = kv.table(2)?.string(0);
          if (v != null) items[key] = v;
        }
      }
      out.add(
        LitertLmSection(
          begin: obj.u64(1),
          end: obj.u64(2),
          dataType: obj.u8(3),
          metadata: items,
        ),
      );
    }
    return out;
  }
}

/// Extracts, for each `(target, dataType, key, value)` job, the matching
/// section of the bundle at [bundlePath] to `target`. Throws if a section is
/// missing. Runs synchronously (call it from an isolate).
void unpackLitertLm(
  String bundlePath,
  List<(String target, int dataType, String key, String value)> jobs,
) {
  final file = LitertLmFile.open(bundlePath);
  for (final (target, dataType, key, value) in jobs) {
    final section = file.find(dataType, key, value);
    if (section == null) {
      throw FormatException('No section with $key=$value in the bundle', bundlePath);
    }
    file.extract(section, target);
  }
}

/// Minimal read-only FlatBuffer table accessor (no generated code): fields
/// are addressed by their index in the schema.
class _Table {
  _Table(this._b, this._data, this._pos);

  factory _Table.root(Uint8List bytes) {
    final b = ByteData.sublistView(bytes);
    return _Table(b, bytes, b.getUint32(0, Endian.little));
  }

  final ByteData _b;
  final Uint8List _data;
  final int _pos;

  /// Absolute offset of field [i], or -1 if it is absent (default value).
  int _field(int i) {
    final vtable = _pos - _b.getInt32(_pos, Endian.little);
    final vtableSize = _b.getUint16(vtable, Endian.little);
    final slot = 4 + 2 * i;
    if (slot >= vtableSize) return -1;
    final off = _b.getUint16(vtable + slot, Endian.little);
    return off == 0 ? -1 : _pos + off;
  }

  /// Follows the u32 forward offset stored at [at].
  int _indirect(int at) => at + _b.getUint32(at, Endian.little);

  int u8(int i) {
    final f = _field(i);
    return f < 0 ? 0 : _b.getUint8(f);
  }

  int u64(int i) {
    final f = _field(i);
    return f < 0 ? 0 : _b.getUint64(f, Endian.little);
  }

  String? string(int i) {
    final f = _field(i);
    if (f < 0) return null;
    final s = _indirect(f);
    final len = _b.getUint32(s, Endian.little);
    return String.fromCharCodes(_data.sublist(s + 4, s + 4 + len));
  }

  _Table? table(int i) {
    final f = _field(i);
    return f < 0 ? null : _Table(_b, _data, _indirect(f));
  }

  /// A vector of tables.
  List<_Table> tables(int i) {
    final f = _field(i);
    if (f < 0) return const [];
    final v = _indirect(f);
    final n = _b.getUint32(v, Endian.little);
    return [for (var k = 0; k < n; k++) _Table(_b, _data, _indirect(v + 4 + 4 * k))];
  }
}
