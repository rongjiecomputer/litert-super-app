import 'dart:convert';
import 'dart:io';

/// Byte-level BPE tokenizer for Qwen2/Qwen3 `tokenizer.json` files.
///
/// Implements the pieces the Qwen3-TTS prompt needs: added (special) token
/// splitting, the Qwen pre-tokenizer regex, GPT-2 byte mapping and ranked
/// BPE merges. Unicode NFC normalization is not applied (input text is
/// assumed to already be NFC, which is the norm for typed text).
class QwenTokenizer {
  QwenTokenizer._(this._vocab, this._ranks, this._added);

  final Map<String, int> _vocab;
  final Map<String, int> _ranks;
  final Map<String, int> _added;
  final Map<String, List<int>> _cache = {};
  late final RegExp _addedRe = RegExp(
    _added.keys.map(RegExp.escape).join('|'),
  );

  static QwenTokenizer fromFile(String path) =>
      fromJson(jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>);

  static QwenTokenizer fromJson(Map<String, dynamic> json) {
    final model = json['model'] as Map<String, dynamic>;
    final vocab = (model['vocab'] as Map<String, dynamic>).map(
      (k, v) => MapEntry(k, v as int),
    );
    final ranks = <String, int>{};
    final merges = model['merges'] as List<dynamic>;
    for (var i = 0; i < merges.length; i++) {
      final m = merges[i];
      final key = m is String ? m : (m as List).join(' ');
      ranks[key] = i;
    }
    final added = <String, int>{};
    for (final t in (json['added_tokens'] as List<dynamic>? ?? const [])) {
      final map = t as Map<String, dynamic>;
      added[map['content'] as String] = map['id'] as int;
    }
    return QwenTokenizer._(vocab, ranks, added);
  }

  int? tokenId(String token) => _added[token] ?? _vocab[token];

  /// Encodes [text] (special tokens like `<|im_start|>` are recognised).
  List<int> encode(String text) {
    final out = <int>[];
    var last = 0;
    if (_added.isNotEmpty) {
      for (final m in _addedRe.allMatches(text)) {
        _encodeOrdinary(text.substring(last, m.start), out);
        out.add(_added[m.group(0)!]!);
        last = m.end;
      }
    }
    _encodeOrdinary(text.substring(last), out);
    return out;
  }

  void _encodeOrdinary(String text, List<int> out) {
    if (text.isEmpty) return;
    for (final m in _pretokenRe.allMatches(text)) {
      final piece = m.group(0)!;
      out.addAll(_cache[piece] ??= _bpe(piece));
    }
  }

  List<int> _bpe(String piece) {
    final bytes = utf8.encode(piece);
    var parts = [for (final b in bytes) _byteToUnicode[b]];
    while (parts.length > 1) {
      var best = -1;
      var bestRank = 1 << 30;
      for (var i = 0; i < parts.length - 1; i++) {
        final r = _ranks['${parts[i]} ${parts[i + 1]}'];
        if (r != null && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      final a = parts[best], b = parts[best + 1];
      final merged = <String>[];
      for (var i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] == a && parts[i + 1] == b) {
          merged.add(a + b);
          i++;
        } else {
          merged.add(parts[i]);
        }
      }
      parts = merged;
    }
    return [
      for (final p in parts)
        _vocab[p] ?? (throw StateError('Token "$p" missing from vocab')),
    ];
  }
}

/// Qwen2 pre-tokenizer pattern. The original uses an inline `(?i:...)`
/// group for contractions; Dart regexps spell the cases out instead.
final _pretokenRe = RegExp(
  r"'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
  r"|[^\r\n\p{L}\p{N}]?\p{L}+"
  r"|\p{N}"
  r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
  r"|\s*[\r\n]+"
  r"|\s+(?!\S)"
  r"|\s+",
  unicode: true,
);

/// GPT-2 byte → printable unicode mapping used by byte-level BPE.
final List<String> _byteToUnicode = () {
  final bs = <int>[
    for (var i = 0x21; i <= 0x7e; i++) i,
    for (var i = 0xa1; i <= 0xac; i++) i,
    for (var i = 0xae; i <= 0xff; i++) i,
  ];
  final cs = [...bs];
  var n = 0;
  for (var b = 0; b < 256; b++) {
    if (!bs.contains(b)) {
      bs.add(b);
      cs.add(256 + n);
      n++;
    }
  }
  final table = List<String>.filled(256, '');
  for (var i = 0; i < bs.length; i++) {
    table[bs[i]] = String.fromCharCode(cs[i]);
  }
  return table;
}();
