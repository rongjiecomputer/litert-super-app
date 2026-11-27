import 'dart:convert';
import 'dart:io';

import 'package:ai_litert/features/tts/text/qwen_tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

import 'tts_test_utils.dart';

void main() {
  final path = modelFile('qwen3-tts-0.6b', 'tokenizer.json');
  test(
    'matches Hugging Face tokenizers ids',
    () {
      final tok = QwenTokenizer.fromFile(path!);
      final cases = jsonDecode(File(golden('qwen_tokenizer.json')).readAsStringSync())
          as List<dynamic>;
      for (final c in cases) {
        final m = c as Map<String, dynamic>;
        expect(tok.encode(m['text'] as String), m['ids'], reason: m['text'] as String);
      }
    },
    skip: path == null ? 'Qwen3-TTS tokenizer.json not downloaded' : false,
  );
}
