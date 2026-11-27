"""Generates small golden fixtures for the Dart TTS unit tests.

    py -I tools/tts/make_goldens.py [--models ~/.litert-lm/models]

Needs `pip install tokenizers numpy librosa`. Writes JSON files to
test/goldens/tts/. Only small, derived data is written (token ids, a few
mel values), never model weights.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'test' / 'goldens' / 'tts'

TOKENIZER_CASES = [
    'Hello, world!',
    "It's a test: we'll see what's up.",
    'The year 2026 has 365 days.',
    '  leading and   multiple spaces\nnew line\r\n',
    '你好，世界！今天天气很好。',
    'こんにちは、元気ですか？',
    '안녕하세요 반갑습니다',
    'Ça va? Größe, señor, déjà vu.',
    'Привет, как дела?',
    'emoji 😀👍 and symbols #$%&*()',
    '<|im_start|>assistant\nHello there.<|im_end|>\n<|im_start|>assistant\n',
]


def tokenizer_goldens(models: Path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(models / 'qwen3-tts-0.6b' / 'tokenizer.json'))
    cases = [{'text': t, 'ids': tok.encode(t).ids} for t in TOKENIZER_CASES]
    (OUT / 'qwen_tokenizer.json').write_text(
        json.dumps(cases, ensure_ascii=False, indent=1), encoding='utf-8')
    print('qwen_tokenizer.json', len(cases))


def mel_goldens():
    """Mel spectrogram of a deterministic chirp, matching Qwen3-TTS's
    speaker-encoder front end (modeling_qwen3_tts.mel_spectrogram)."""
    import librosa
    sr = 24000
    n = sr // 2
    t = np.arange(n) / sr
    y = (0.5 * np.sin(2 * math.pi * (200 + 1800 * t) * t)
         + 0.1 * np.sin(2 * math.pi * 3000 * t)).astype(np.float32)
    n_fft, hop, win, n_mels, fmin, fmax = 1024, 256, 1024, 128, 0, 12000
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                    fmin=fmin, fmax=fmax)
    pad = (n_fft - hop) // 2
    yp = np.pad(y, (pad, pad), mode='reflect')
    window = np.hanning(win + 1)[:-1].astype(np.float64)  # periodic hann
    frames = 1 + (len(yp) - n_fft) // hop
    spec = np.empty((n_fft // 2 + 1, frames))
    for i in range(frames):
        seg = yp[i * hop:i * hop + n_fft] * window
        f = np.fft.rfft(seg)
        spec[:, i] = np.sqrt(f.real ** 2 + f.imag ** 2 + 1e-9)
    mel = np.log(np.clip(mel_basis @ spec, 1e-5, None))  # [128, T]
    data = {
        'sr': sr, 'signal': {'f0': 200, 'sweep': 1800, 'tone': 3000,
                             'n': n},
        'frames': frames,
        # Sampled positions keep the fixture small.
        'samples': [
            {'mel': int(m), 'frame': int(f), 'value': float(mel[m, f])}
            for m in range(0, n_mels, 7) for f in range(0, frames, 5)
        ],
        'filter_row_sums': [float(s) for s in mel_basis.sum(axis=1)],
    }
    (OUT / 'qwen_mel.json').write_text(json.dumps(data), encoding='utf-8')
    print('qwen_mel.json', frames, 'frames')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models',
                    default=os.path.expanduser('~/.litert-lm/models'))
    ap.add_argument('--only', choices=['tokenizer', 'mel'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, 'tokenizer'):
        tokenizer_goldens(Path(a.models))
    if a.only in (None, 'mel'):
        mel_goldens()


if __name__ == '__main__':
    main()
