"""Exports the Qwen3-TTS speaker encoder (ECAPA-TDNN x-vector) to LiteRT.

The litert-community Qwen3-TTS conversion clones voices from a 1024-d
speaker embedding but does not ship the encoder that computes it. This
script builds `speaker_encoder.tflite` so the app can clone a voice from a
reference recording entirely on device.

Run on Linux or macOS (litert-torch has no Windows wheels; on Windows use
WSL):

    python3 -m venv ~/litert-venv
    ~/litert-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch<2.15'
    ~/litert-venv/bin/pip install litert-torch safetensors numpy requests
    ~/litert-venv/bin/python tools/tts/export_qwen3_speaker_encoder.py \
        --out ~/.litert-lm/models/qwen3-tts-0.6b/speaker_encoder.tflite

Only the encoder's weights (~18 MB) are fetched from
`Qwen/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors` with HTTP range requests,
so the full 1.8 GB checkpoint is not downloaded (pass --safetensors to use
a local copy instead).

The graph has one signature per fixed input length (litert-torch cannot
keep the time axis dynamic): `frames_<T>` takes a log-mel spectrogram
`[1, T, 128]` and returns the x-vector `[1, 1024]`. The app crops or tiles
the reference audio to the nearest bucket.

The encoder modules below are copied from QwenLM/Qwen3-TTS
(qwen_tts/core/models/modeling_qwen3_tts.py, Apache-2.0) without their
transformers dependencies.
"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
PREFIX = 'speaker_encoder.'
SAMPLE_RATE = 24000
HOP = 256
N_FFT = 1024
# Reference lengths in seconds -> mel frames (center=False, reflect pad 384).
BUCKET_SECONDS = (3, 6, 10)
ROOT = Path(__file__).resolve().parents[2]


def frames_for(seconds):
    n = seconds * SAMPLE_RATE
    return 1 + (n + 2 * ((N_FFT - HOP) // 2) - N_FFT) // HOP


# ---- Encoder (verbatim structure from modeling_qwen3_tts.py) ---------------

class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding='same',
                              padding_mode='reflect')
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3,
                 dilation=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(in_channels // scale, out_channels // scale,
                              kernel_size=kernel_size, dilation=dilation)
            for _ in range(scale - 1)])
        self.scale = scale

    def forward(self, x):
        outputs = []
        out = None
        for i, part in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                out = part
            elif i == 1:
                out = self.blocks[i - 1](part)
            else:
                out = self.blocks[i - 1](part + out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, 1, padding='same',
                               padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, 1, padding='same',
                               padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m = x.mean(dim=2, keepdim=True)
        m = self.sigmoid(self.conv2(self.relu(self.conv1(m))))
        return x * m


class AttentiveStatisticsPooling(nn.Module):
    """ASP for a single, unpadded utterance (the reference builds an
    all-ones mask, so masking is dropped here; results are identical)."""

    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, 1,
                              padding='same', padding_mode='reflect')

    def _stats(self, x, w):
        mean = (w * x).sum(2)
        std = torch.sqrt(
            (w * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(self.eps))
        return mean, std

    def forward(self, x):
        t = x.shape[-1]
        mean, std = self._stats(x, torch.full_like(x[:, :1], 1.0 / t))
        attn_in = torch.cat([x, mean.unsqueeze(2).expand(-1, -1, t),
                             std.unsqueeze(2).expand(-1, -1, t)], dim=1)
        attn = F.softmax(self.conv(self.tanh(self.tdnn(attn_in))), dim=2)
        mean, std = self._stats(x, attn)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8,
                 se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, 1, 1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels,
                                          res2net_scale, kernel_size,
                                          dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, 1, 1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels,
                                               out_channels)

    def forward(self, x):
        y = self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x))))
        return y + x


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch, ks, dil = (cfg['enc_channels'], cfg['enc_kernel_sizes'],
                       cfg['enc_dilations'])
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(cfg['mel_dim'], ch[0], ks[0], dil[0])])
        for i in range(1, len(ch) - 1):
            self.blocks.append(SqueezeExcitationRes2NetBlock(
                ch[i - 1], ch[i], res2net_scale=cfg['enc_res2net_scale'],
                se_channels=cfg['enc_se_channels'], kernel_size=ks[i],
                dilation=dil[i]))
        self.mfa = TimeDelayNetBlock(ch[-1], ch[-1], ks[-1], dil[-1])
        self.asp = AttentiveStatisticsPooling(
            ch[-1], attention_channels=cfg['enc_attention_channels'])
        self.fc = nn.Conv1d(ch[-1] * 2, cfg['enc_dim'], 1, padding='same',
                            padding_mode='reflect')

    def forward(self, mel):  # mel: [1, T, 128]
        x = mel.transpose(1, 2)
        feats = []
        for layer in self.blocks:
            x = layer(x)
            feats.append(x)
        x = self.mfa(torch.cat(feats[1:], dim=1))
        return self.fc(self.asp(x)).squeeze(-1)  # [1, 1024]


DEFAULT_CFG = dict(mel_dim=128, enc_dim=1024,
                   enc_channels=[512, 512, 512, 512, 1536],
                   enc_kernel_sizes=[5, 3, 3, 3, 1],
                   enc_dilations=[1, 2, 3, 4, 1], enc_attention_channels=128,
                   enc_res2net_scale=8, enc_se_channels=128)


# ---- Weights ---------------------------------------------------------------

_DTYPES = {'BF16': (2, None), 'F16': (2, np.float16), 'F32': (4, np.float32)}


def _decode(raw, dtype, shape):
    if dtype == 'BF16':
        u16 = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
        return u16.view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=_DTYPES[dtype][1]).astype(
        np.float32).reshape(shape)


def load_weights(safetensors=None):
    """Returns ({name: np.ndarray}, config) for the speaker encoder."""
    if safetensors:
        data = Path(safetensors).read_bytes()
        n = struct.unpack('<Q', data[:8])[0]
        header = json.loads(data[8:8 + n])
        read = lambda a, b: data[8 + n + a:8 + n + b]  # noqa: E731
        cfg_path = Path(safetensors).with_name('config.json')
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    else:
        import requests
        url = f'https://huggingface.co/{REPO}/resolve/main/model.safetensors'
        s = requests.Session()

        def get(a, b):  # inclusive byte range
            r = s.get(url, headers={'Range': f'bytes={a}-{b}'}, timeout=120)
            r.raise_for_status()
            if len(r.content) != b - a + 1:
                raise RuntimeError('Server ignored the Range request')
            return r.content

        n = struct.unpack('<Q', get(0, 7))[0]
        header = json.loads(get(8, 8 + n - 1))
        keys = [k for k in header if k.startswith(PREFIX)]
        lo = min(header[k]['data_offsets'][0] for k in keys)
        hi = max(header[k]['data_offsets'][1] for k in keys)
        print(f'fetching {len(keys)} tensors ({(hi - lo) / 1e6:.1f} MB)')
        blob = get(8 + n + lo, 8 + n + hi - 1)
        read = lambda a, b: blob[a - lo:b - lo]  # noqa: E731
        config = s.get(f'https://huggingface.co/{REPO}/resolve/main/'
                       'config.json', timeout=60).json()
    weights = {}
    for k, v in header.items():
        if not k.startswith(PREFIX):
            continue
        a, b = v['data_offsets']
        weights[k[len(PREFIX):]] = _decode(read(a, b), v['dtype'], v['shape'])
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in config.get('speaker_encoder_config',
                                            {}).items() if k in cfg})
    return weights, cfg


# ---- Reference mel (modeling_qwen3_tts.mel_spectrogram) -------------------

def mel_spectrogram(y):
    import librosa
    mel = torch.from_numpy(librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=128, fmin=0, fmax=12000)).float()
    pad = (N_FFT - HOP) // 2
    y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    spec = torch.stft(y, N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=torch.hann_window(N_FFT), center=False,
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(mel @ spec, min=1e-5)).transpose(1, 2)


def test_signal(seconds=4.0):
    """Deterministic voiced-like test signal (regenerated bit-for-bit-ish by
    the Dart parity test): harmonic stack with vibrato and tremolo."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 140 + 25 * np.sin(2 * math.pi * 3 * t)
    phase = 2 * math.pi * np.cumsum(f0) / SAMPLE_RATE
    y = sum(0.3 / h * np.sin(h * phase) for h in range(1, 9))
    y = y * (0.6 + 0.4 * np.sin(2 * math.pi * 0.5 * t))
    return y.astype(np.float32)


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='speaker_encoder.tflite')
    ap.add_argument('--safetensors', help='local model.safetensors')
    ap.add_argument('--fixture', default=str(
        ROOT / 'test' / 'goldens' / 'tts' / 'speaker_encoder_fixture.json'))
    a = ap.parse_args()

    weights, cfg = load_weights(a.safetensors)
    model = SpeakerEncoder(cfg).eval()
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in weights.items()},
        strict=False)
    if missing or unexpected:
        raise SystemExit(f'weight mismatch: missing={missing} '
                         f'unexpected={unexpected}')

    import litert_torch
    buckets = [frames_for(s) for s in BUCKET_SECONDS]
    builder = None
    for t in buckets:
        args = (torch.randn(1, t, 128),)
        builder = (litert_torch.signature(f'frames_{t}', model, args)
                   if builder is None else
                   builder.signature(f'frames_{t}', model, args))
    edge = builder.convert()
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    edge.export(str(out))
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB), '
          f'signatures {[f"frames_{t}" for t in buckets]}')

    # Parity: torch vs LiteRT on the deterministic test signal.
    y = test_signal()
    mel = mel_spectrogram(torch.from_numpy(y)[None])
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(out))
    worst = 1.0
    for t in buckets:
        m = mel[:, :t] if mel.shape[1] >= t else mel.repeat(
            1, math.ceil(t / mel.shape[1]), 1)[:, :t]
        with torch.no_grad():
            ref = model(m).numpy()
        runner = interp.get_signature_runner(f'frames_{t}')
        name = next(iter(runner.get_input_details()))
        got = next(iter(runner(**{name: m.numpy()}).values()))
        c = cosine(ref, got)
        worst = min(worst, c)
        print(f'frames_{t}: cosine(torch, litert) = {c:.6f}')
    if worst < 0.999:
        raise SystemExit('LiteRT output does not match PyTorch')

    # Fixture for the Dart integration test (mel + x-vector of the first
    # bucket, computed with the reference front end).
    t0 = buckets[0]
    with torch.no_grad():
        ref = model(mel[:, :t0]).numpy()[0]
    fixture = {
        'seconds': 4.0, 'frames': t0,
        'mel_samples': [[int(f), int(c), float(mel[0, f, c])]
                        for f in range(0, t0, 37) for c in range(0, 128, 11)],
        'xvector': [float(v) for v in ref],
    }
    Path(a.fixture).parent.mkdir(parents=True, exist_ok=True)
    Path(a.fixture).write_text(json.dumps(fixture))
    print('wrote', a.fixture)


if __name__ == '__main__':
    main()
