import 'package:ai_litert/features/tts/text/english_g2p.dart';
import 'package:flutter_test/flutter_test.dart';

import 'tts_test_utils.dart';

void main() {
  group('EnglishTextNormalizer', () {
    String n(String s) => EnglishTextNormalizer.normalize(s);
    test('numbers, currency, percent, ordinals, years', () {
      expect(n('I have 42 cats'), 'I have forty two cats');
      expect(n(r'It costs $5.99'), 'It costs five dollars and ninety nine cents');
      expect(n('100% sure'), 'one hundred percent sure');
      expect(n('the 3rd time'), 'the third time');
      expect(n('in 1984 and 2007'), 'in nineteen eighty four and two thousand seven');
      expect(n('pi is 3.14'), 'pi is three point one four');
      expect(n('covid19'), 'covid nineteen');
      expect(n('1,250,000 people'), 'one million two hundred fifty thousand people');
      expect(n('A & B'), 'A and B');
      expect(n('José señor'), 'Jose senor');
    });
  });

  test('splitSentences keeps punctuation and splits long text', () {
    expect(
      splitSentences('Hello there. How are you? Fine!'),
      ['Hello there.', 'How are you?', 'Fine!'],
    );
    expect(splitSentences('你好。今天好吗？'), ['你好。', '今天好吗？']);
    final long = List.filled(60, 'word').join(' ');
    final parts = splitSentences(long, maxChars: 100);
    expect(parts.every((p) => p.length <= 101), isTrue);
    expect(parts.join(' ').split(' ').length, 60);
  });

  final cmu = modelFile('kokoro-82m', 'cmudict.txt');
  final config = modelFile('kokoro-82m', 'config.json');
  test(
    'CMU lookup produces misaki-style IPA',
    () {
      final dict = EnglishPhonemizer.loadCmu(cmu!);
      final vocab = EnglishPhonemizer.loadVocab(config!);
      final g2p = EnglishPhonemizer(dict, vocab);
      expect(g2p.phonemeString('Hello world.'), 'həlˈO wˈɜɹld.');
      expect(g2p.phonemeString('The cat sat'), 'ðə kˈæt sˈæt');
      // Unknown word without a neural fallback is spelled out.
      expect(g2p.phonemeString('xq'), 'ˈɛks kjˈu');
      final gb = EnglishPhonemizer(dict, vocab, british: true);
      expect(gb.phonemeString('go'), 'ɡˈQ');
    },
    skip: cmu == null || config == null ? 'Kokoro not downloaded' : false,
  );
}
