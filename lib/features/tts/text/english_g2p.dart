import 'dart:convert';
import 'dart:io';

/// Neural fallback for out-of-dictionary words: returns stress-less
/// ARPABET symbols (e.g. `[HH, AH, L, OW]`) or an empty list.
typedef ArpabetFallback = List<String> Function(String word);

/// English text → Kokoro phoneme token ids.
///
/// Pipeline (ported from the LiteRT-Models Kokoro sample, MIT):
/// 1. [EnglishTextNormalizer] spells out numbers, currency and symbols.
/// 2. In-dictionary words use the CMU Pronouncing Dictionary (with stress),
///    mapped ARPABET → misaki IPA (Kokoro's phoneme alphabet).
/// 3. Out-of-dictionary words use the neural G2P [fallback] (DeepPhonemizer
///    en_us_cmudict_forward on LiteRT), else are spelled out letter by letter.
///
/// No POS context is used, so homographs take their first CMU entry.
class EnglishPhonemizer {
  EnglishPhonemizer(this.cmu, this.vocab, {this.fallback, this.british = false});

  final Map<String, String> cmu;
  final Map<String, int> vocab;
  final ArpabetFallback? fallback;

  /// GB voices: a few vowel tweaks so `bf_`/`bm_` voices sound less American.
  final bool british;

  int get spaceId => vocab[' ']!;

  static Map<String, String> loadCmu(String path) {
    final out = <String, String>{};
    for (final line in File(path).readAsLinesSync()) {
      final sp = line.indexOf(' ');
      if (sp <= 0) continue;
      out.putIfAbsent(line.substring(0, sp), () => line.substring(sp + 1));
    }
    return out;
  }

  /// Kokoro `config.json` → phoneme vocabulary.
  static Map<String, int> loadVocab(String configPath) {
    final json = jsonDecode(File(configPath).readAsStringSync()) as Map<String, dynamic>;
    return (json['vocab'] as Map<String, dynamic>).map((k, v) => MapEntry(k, v as int));
  }

  /// Phoneme ids for [text] (one sentence or more), without BOS/EOS.
  List<int> phonemize(String text) {
    final out = <int>[];
    final tokens = _tokenize(EnglishTextNormalizer.normalize(text));
    var prevWasWord = false;
    for (final tok in tokens) {
      final punct = vocab[tok];
      if (tok.length == 1 && _punct.contains(tok) && punct != null) {
        // Drop leading punctuation and duplicates; keep sentence marks.
        if (out.isNotEmpty && out.last != punct) out.add(punct);
        prevWasWord = false;
        continue;
      }
      if (prevWasWord) out.add(spaceId);
      prevWasWord = true;
      _word(tok, out);
    }
    return out;
  }

  /// Phoneme string for [text] (debugging / tests).
  String phonemeString(String text) {
    final inv = {for (final e in vocab.entries) e.value: e.key};
    return phonemize(text).map((i) => inv[i] ?? '?').join();
  }

  void _word(String word, List<int> out) {
    final lower = word.toLowerCase();
    final arpa = cmu[lower];
    if (arpa != null) {
      _arpabet(arpa.split(' '), out);
      return;
    }
    // Hyphenated / apostrophe compounds: try the parts.
    if (lower.contains('-')) {
      final parts = lower.split('-').where((p) => p.isNotEmpty).toList();
      for (var i = 0; i < parts.length; i++) {
        if (i > 0) out.add(spaceId);
        _word(parts[i], out);
      }
      return;
    }
    final fb = fallback?.call(lower.replaceAll(RegExp("[^a-z']"), ''));
    if (fb != null && fb.isNotEmpty) {
      // The neural model is stress-less: stress the first vowel so the
      // word isn't read flat.
      final syms = <String>[];
      var stressed = false;
      for (final s in fb) {
        if (!stressed && _vowels.contains(s)) {
          syms.add('${s}1');
          stressed = true;
        } else {
          syms.add(s);
        }
      }
      _arpabet(syms, out);
      return;
    }
    _spellOut(lower, out);
  }

  void _spellOut(String word, List<int> out) {
    var first = true;
    for (final c in word.split('')) {
      final arpa = _letters[c] ?? (_digitWords[c] != null ? cmu[_digitWords[c]] : null);
      if (arpa == null) continue;
      if (!first) out.add(spaceId);
      first = false;
      _arpabet(arpa.split(' '), out);
    }
  }

  void _arpabet(List<String> syms, List<int> out) {
    for (final sym in syms) {
      if (sym.isEmpty) continue;
      final last = sym.codeUnitAt(sym.length - 1);
      final hasStress = last >= 0x30 && last <= 0x39;
      final stress = hasStress ? last - 0x30 : -1;
      final base = hasStress ? sym.substring(0, sym.length - 1) : sym;
      final ipa = _ipa(base, stress);
      if (ipa == null) continue;
      if (stress == 1) {
        out.add(vocab['ˈ']!);
      } else if (stress == 2) {
        out.add(vocab['ˌ']!);
      }
      for (final ch in ipa.runes) {
        final id = vocab[String.fromCharCode(ch)];
        if (id != null) out.add(id);
      }
    }
  }

  String? _ipa(String sym, int stress) => switch (sym) {
    'AA' => 'ɑ',
    'AE' => 'æ',
    'AH' => stress >= 1 ? 'ʌ' : 'ə',
    'AO' => 'ɔ',
    'AW' => 'W', // misaki /aʊ/
    'AY' => 'I', // misaki /aɪ/
    'B' => 'b',
    'CH' => 'ʧ',
    'D' => 'd',
    'DH' => 'ð',
    'EH' => 'ɛ',
    'ER' => stress >= 1 ? (british ? 'ɜː' : 'ɜɹ') : (british ? 'ə' : 'ɚ'),
    'EY' => 'A', // misaki /eɪ/
    'F' => 'f',
    'G' => 'ɡ',
    'HH' => 'h',
    'IH' => 'ɪ',
    'IY' => 'i',
    'JH' => 'ʤ',
    'K' => 'k',
    'L' => 'l',
    'M' => 'm',
    'N' => 'n',
    'NG' => 'ŋ',
    'OW' => british ? 'Q' : 'O', // misaki /oʊ/ (US) vs /əʊ/ (GB)
    'OY' => 'Y', // misaki /ɔɪ/
    'P' => 'p',
    'R' => 'ɹ',
    'S' => 's',
    'SH' => 'ʃ',
    'T' => 't',
    'TH' => 'θ',
    'UH' => 'ʊ',
    'UW' => 'u',
    'V' => 'v',
    'W' => 'w',
    'Y' => 'j',
    'Z' => 'z',
    'ZH' => 'ʒ',
    _ => null,
  };

  static List<String> _tokenize(String text) {
    final tokens = <String>[];
    final sb = StringBuffer();
    void flush() {
      if (sb.isNotEmpty) {
        tokens.add(sb.toString());
        sb.clear();
      }
    }

    for (final r in text.runes) {
      final c = String.fromCharCode(r);
      final isWordChar = _isLetterOrDigit(r) || c == "'" || c == '-';
      if (isWordChar) {
        sb.write(c);
      } else {
        flush();
        if (_punct.contains(c)) tokens.add(c);
      }
    }
    flush();
    // Trim stray apostrophes/hyphens at word edges ("'quoted'", "--").
    return [
      for (final t in tokens)
        if (_punct.contains(t)) t else t.replaceAll(RegExp(r"^['-]+|['-]+$"), ''),
    ].where((t) => t.isNotEmpty).toList();
  }

  static bool _isLetterOrDigit(int r) =>
      (r >= 0x30 && r <= 0x39) ||
      (r >= 0x41 && r <= 0x5a) ||
      (r >= 0x61 && r <= 0x7a) ||
      (r >= 0xc0 && r <= 0x24f && r != 0xd7 && r != 0xf7);

  static const _punct = {'.', ',', '!', '?', ';', ':', '—', '…'};
  static const _vowels = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', //
    'OY', 'UH', 'UW',
  };
  static const _digitWords = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', //
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
  };
  static const _letters = {
    'a': 'EY1', 'b': 'B IY1', 'c': 'S IY1', 'd': 'D IY1', 'e': 'IY1', //
    'f': 'EH1 F', 'g': 'JH IY1', 'h': 'EY1 CH', 'i': 'AY1', 'j': 'JH EY1',
    'k': 'K EY1', 'l': 'EH1 L', 'm': 'EH1 M', 'n': 'EH1 N', 'o': 'OW1',
    'p': 'P IY1', 'q': 'K Y UW1', 'r': 'AA1 R', 's': 'EH1 S', 't': 'T IY1',
    'u': 'Y UW1', 'v': 'V IY1', 'w': 'D AH1 B AH0 L Y UW0', 'x': 'EH1 K S',
    'y': 'W AY1', 'z': 'Z IY1',
  };
}

/// Spells out numbers, currency and symbols so they aren't dropped.
abstract final class EnglishTextNormalizer {
  static String normalize(String text) {
    var t = asciiFold(text);
    t = t.replaceAll('...', '…').replaceAll(RegExp(r'\s+[-–]\s+|--'), ' — ');
    // Ordinals before splitting letters from digits ("3rd" → "third").
    t = t.replaceAllMapped(
      RegExp(r'\b(\d+)(st|nd|rd|th)\b', caseSensitive: false),
      (m) => ordinal(int.parse(m[1]!)),
    );
    t = t.replaceAllMapped(
      RegExp(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])'),
      (_) => ' ',
    );
    t = t.replaceAllMapped(RegExp(r'\$\s?(\d+(?:,\d{3})*)(?:\.(\d{1,2}))?'), (m) {
      final dollars = cardinal(int.parse(m[1]!.replaceAll(',', '')));
      final cents = m[2];
      if (cents != null && cents.isNotEmpty) {
        final c = int.parse(cents.padRight(2, '0'));
        return '$dollars dollars and ${cardinal(c)} cents';
      }
      return '$dollars dollars';
    });
    t = t.replaceAllMapped(
      RegExp(r'(\d+(?:\.\d+)?)\s?%'),
      (m) => '${spokenNumber(m[1]!)} percent',
    );
    // Years read as pairs ("1984" → "nineteen eighty four").
    t = t.replaceAllMapped(RegExp(r'\b(1[1-9]|20)(\d\d)\b(?![.,]\d)'), (m) {
      final hi = int.parse(m[1]!), lo = int.parse(m[2]!);
      if (hi == 20 && lo < 10) return cardinal(hi * 100 + lo);
      return '${cardinal(hi)} ${lo == 0 ? 'hundred' : lo < 10 ? 'oh ${cardinal(lo)}' : cardinal(lo)}';
    });
    t = t.replaceAllMapped(
      RegExp(r'\d+(?:,\d{3})*(?:\.\d+)?'),
      (m) => spokenNumber(m[0]!),
    );
    for (final e in _symbols.entries) {
      t = t.replaceAll(e.key, e.value);
    }
    return t.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  static String spokenNumber(String s) {
    final clean = s.replaceAll(',', '');
    final dot = clean.indexOf('.');
    if (dot < 0) return cardinal(int.parse(clean));
    final ip = dot == 0 ? 0 : int.parse(clean.substring(0, dot));
    final frac = clean.substring(dot + 1).split('').map((d) => _ones[int.parse(d)]).join(' ');
    return '${cardinal(ip)} point $frac';
  }

  static String cardinal(int n) {
    if (n == 0) return 'zero';
    if (n < 0) return 'minus ${cardinal(-n)}';
    final parts = <String>[];
    var rem = n;
    for (final (value, name) in _scales) {
      if (rem >= value) {
        parts.add('${_three(rem ~/ value)} $name');
        rem %= value;
      }
    }
    if (rem > 0) parts.add(_three(rem));
    return parts.join(' ');
  }

  static String ordinal(int n) {
    final words = cardinal(n).split(' ');
    final last = words.removeLast();
    final ord = _ordinalWords[last] ??
        (last.endsWith('y') ? '${last.substring(0, last.length - 1)}ieth' : '${last}th');
    return [...words, ord].join(' ');
  }

  static String _three(int n) {
    final h = n ~/ 100, t = n % 100;
    final parts = <String>[];
    if (h > 0) parts.add('${_ones[h]} hundred');
    if (t >= 1 && t <= 19) {
      parts.add(_ones[t]);
    } else if (t >= 20) {
      parts.add(t % 10 == 0 ? _tens[t ~/ 10] : '${_tens[t ~/ 10]} ${_ones[t % 10]}');
    }
    return parts.join(' ');
  }

  /// Strips diacritics from common Latin letters (é → e, ñ → n).
  static String asciiFold(String s) {
    final sb = StringBuffer();
    for (final r in s.runes) {
      final c = String.fromCharCode(r);
      sb.write(_fold[c] ?? c);
    }
    return sb.toString();
  }

  static const _ones = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', //
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen',
  ];
  static const _tens = [
    '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', //
    'eighty', 'ninety',
  ];
  static const _scales = [
    (1000000000000, 'trillion'),
    (1000000000, 'billion'),
    (1000000, 'million'),
    (1000, 'thousand'),
  ];
  static const _ordinalWords = {
    'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth', //
    'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth',
  };
  static const _symbols = {
    '&': ' and ', '%': ' percent ', '@': ' at ', '#': ' number ', //
    '+': ' plus ', '=': ' equals ', '°': ' degrees ', '€': ' euros ',
    '£': ' pounds ',
  };
  static const _fold = {
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'ā': 'a', //
    'ç': 'c', 'č': 'c', 'ć': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ē': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ñ': 'n', 'ń': 'n',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'ō': 'o',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ū': 'u', 'ý': 'y', 'ÿ': 'y',
    'š': 's', 'ś': 's', 'ž': 'z', 'ź': 'z', 'ż': 'z', 'ł': 'l', 'ß': 'ss',
    'æ': 'ae', 'œ': 'oe', 'À': 'A', 'Á': 'A', 'Â': 'A', 'Ä': 'A', 'Å': 'A',
    'Ç': 'C', 'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E', 'Í': 'I', 'Ñ': 'N',
    'Ó': 'O', 'Ö': 'O', 'Ø': 'O', 'Ú': 'U', 'Ü': 'U', 'Š': 'S', 'Ž': 'Z',
    '‘': "'", '’': "'", '“': '"', '”': '"', '–': '-',
  };
}

/// Splits text into sentences on `.!?…` (and CJK `。！？`), keeping the
/// punctuation; long sentences are further split at `,;:` or spaces so no
/// piece exceeds [maxChars].
List<String> splitSentences(String text, {int maxChars = 220}) {
  final parts = <String>[];
  final re = RegExp(r'[^.!?…。！？\n]+(?:[.!?…。！？]+["”’)\]]*|\n|$)');
  for (final m in re.allMatches(text)) {
    final s = m[0]!.trim();
    if (s.isEmpty) continue;
    if (s.length <= maxChars) {
      parts.add(s);
      continue;
    }
    // Break overly long sentences at clause marks, then at spaces.
    var rest = s;
    while (rest.length > maxChars) {
      var cut = rest.lastIndexOf(RegExp(r'[,;:，；、]'), maxChars);
      if (cut < maxChars ~/ 3) cut = rest.lastIndexOf(' ', maxChars);
      if (cut < maxChars ~/ 3) cut = maxChars;
      parts.add(rest.substring(0, cut + 1).trim());
      rest = rest.substring(cut + 1).trim();
    }
    if (rest.isNotEmpty) parts.add(rest);
  }
  return parts;
}
