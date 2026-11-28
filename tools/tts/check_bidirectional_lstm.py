"""Checks native/bidirectional_lstm.c against a NumPy reference.

    py -I -m venv .venv-tools && .venv-tools/Scripts/pip install numpy  # once
    .venv-tools/Scripts/python tools/tts/check_bidirectional_lstm.py [DLL_GLOB]

DLL_GLOB defaults to the library the Dart build hook produced under
`.dart_tool/hooks_runner` (run `flutter test` or `flutter run` first). The
kernel only calls LiteRT through four function pointers, so this script loads
it with ctypes and hands it stand-in tensor buffers; no LiteRT runtime needed.
The reference mirrors LiteRT's own bidirectional_lstm kernel (gate order ifgo,
length-aware reverse gather, padding masked on both halves).
"""
import ctypes as C, glob, sys, numpy as np

dll = glob.glob(sys.argv[1] if len(sys.argv) > 1 else
                '.dart_tool/hooks_runner/shared/ai_litert/build/*/litert_custom_ops.*')[0]
dll = [d for d in glob.glob(dll) if d.endswith(('.dll', '.so', '.dylib'))][0] if not dll.endswith(('.dll', '.so', '.dylib')) else dll
lib = C.CDLL(dll)

class Layout(C.Structure):
    _fields_ = [('rank_bits', C.c_uint32), ('dimensions', C.c_int32*8), ('strides', C.c_uint32*8)]
class Ranked(C.Structure):
    _fields_ = [('element_type', C.c_int), ('layout', Layout)]
assert C.sizeof(Layout) == 68 and C.sizeof(Ranked) == 72

VP = C.c_void_p
INIT = C.CFUNCTYPE(C.c_int, VP, VP, C.c_size_t)
GETL = C.CFUNCTYPE(C.c_int, VP, C.c_size_t, C.POINTER(Layout), C.c_size_t, C.POINTER(Layout))
RUN = C.CFUNCTYPE(C.c_int, VP, C.c_size_t, C.POINTER(VP), C.c_size_t, C.POINTER(VP))
DEST = C.CFUNCTYPE(C.c_int, VP)
class Kernel(C.Structure):
    _fields_ = [('Init', INIT), ('GetOutputLayouts', GETL), ('Run', RUN), ('Destroy', DEST)]

captured = {}
ADD = C.CFUNCTYPE(C.c_int, VP, C.c_char_p, C.c_int, C.POINTER(Kernel), VP)
@ADD
def add_kernel(opts, name, ver, kern, ud):
    captured['name'] = name; captured['ver'] = ver; captured['kernel'] = kern
    return 0

bufs = {}
GETT = C.CFUNCTYPE(C.c_int, VP, C.POINTER(Ranked))
@GETT
def get_type(h, out):
    arr = bufs[h]
    out[0].element_type = 0
    out[0].layout.rank_bits = len(arr.shape)
    for i, d in enumerate(arr.shape): out[0].layout.dimensions[i] = d
    return 0
LOCK = C.CFUNCTYPE(C.c_int, VP, C.POINTER(VP), C.c_int)
@LOCK
def lock(h, out, mode):
    out[0] = bufs[h].ctypes.data; return 0
UNL = C.CFUNCTYPE(C.c_int, VP)
@UNL
def unlock(h): return 0

class Api(C.Structure):
    _fields_ = [('add', ADD), ('get', GETT), ('lock', LOCK), ('unlock', UNL)]
api = Api(add_kernel, get_type, lock, unlock)
reg = lib.litert_custom_ops_register_bidirectional_lstm
reg.argtypes = [VP, C.POINTER(Api)]
assert reg(1, C.byref(api)) == 0
print('registered', captured['name'], captured['ver'])
k = captured['kernel'][0]

def sig(x): return 1/(1+np.exp(-x))
def lstm(x, wih, whh, b):
    T, H = x.shape[0], whh.shape[1]
    h = np.zeros(H); c = np.zeros(H); out = np.zeros((T, H))
    for t in range(T):
        g = wih @ x[t] + whh @ h + b
        i, f, gg, o = g[:H], g[H:2*H], g[2*H:3*H], g[3*H:]
        c = sig(f)*c + sig(i)*np.tanh(gg); h = sig(o)*np.tanh(c); out[t] = h
    return out

def ref(x, L, mask, wif, whf, bf, wib, whb, bb):
    B, T, D = x.shape; H = whf.shape[1]
    y = np.zeros((B, T, 2*H))
    for b in range(B):
        n = min(max(L[b], 1), T)
        idx = np.clip(L[b]-1-np.arange(T), 0, T-1)
        keep = (1-mask[b])[:, None]
        fwd = np.zeros((T, H)); fwd[:n] = lstm(x[b, :n], wif, whf, bf)
        rin = x[b][idx][:n] * keep[:n]
        rout = np.zeros((T, H)); rout[:n] = lstm(rin, wib, whb, bb)
        y[b, :, :H] = fwd * keep
        y[b, :, H:] = rout[idx] * keep
    return y

rng = np.random.default_rng(0)
worst = 0
for (B, T, D, H, L) in [(1, 128, 640, 256, [37]), (1, 128, 512, 256, [128]), (2, 20, 24, 16, [20, 7]), (1, 16, 8, 4, [0])]:
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    mask = np.zeros((B, T), np.float32)
    for b in range(B): mask[b, L[b]:] = 1
    Ls = np.array(L, np.int32)
    w = lambda *s: (rng.standard_normal(s) * 0.15).astype(np.float32)
    wif, whf, bf, wib, whb, bb = w(4*H, D), w(4*H, H), w(4*H), w(4*H, D), w(4*H, H), w(4*H)
    ins = [x, Ls, mask, wif, whf, bf, wib, whb, bb]
    y = np.zeros((B, T, 2*H), np.float32)
    handles = (VP*9)(*range(100, 109))
    for h, a in zip(range(100, 109), ins): bufs[h] = a
    bufs[200] = y
    outs = (VP*1)(200)
    # output layout check
    il = (Layout*9)()
    for i, a in enumerate(ins):
        il[i].rank_bits = a.ndim
        for j, d in enumerate(a.shape): il[i].dimensions[j] = d
    ol = (Layout*1)()
    assert k.GetOutputLayouts(None, 9, il, 1, ol) == 0
    assert (ol[0].rank_bits & 0x7f) == 3 and list(ol[0].dimensions[:3]) == [B, T, 2*H], list(ol[0].dimensions[:3])
    assert k.Run(None, 9, handles, 1, outs) == 0
    r = ref(x.astype(np.float64), L, mask, *[a.astype(np.float64) for a in ins[3:]])
    err = np.abs(y - r).max(); worst = max(worst, err)
    print(f'B={B} T={T} D={D} H={H} len={L}: max abs err {err:.2e}, ref max {np.abs(r).max():.3f}')
assert worst < 1e-4, worst
print('PASS')
