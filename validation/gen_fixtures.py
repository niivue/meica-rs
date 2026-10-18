#!/usr/bin/env python3
"""Write the numpy/scipy/mpmath reference vectors the core's unit tests read.

Usage: gen_fixtures.py OUT_DIR. Deterministic: rerunning gives identical files.
"""
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy import linalg, special, stats

mp.mp.dps = 50
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261017)


def dump(name, obj):
    (out / f'{name}.json').write_text(json.dumps(obj, separators=(',', ':')) + '\n')


def floats(a):
    return [float(v) for v in np.ravel(a)]


def mat(a):
    a = np.asarray(a, dtype=float)
    return {'rows': a.shape[0], 'cols': a.shape[1], 'data': floats(a)}  # row-major


def against(fn_mp, fn_np, xs):
    xs = [float(x) for x in xs]
    return {'x': xs, 'mp': [float(fn_mp(mp.mpf(x))) for x in xs], 'np': floats(fn_np(np.array(xs)))}


# --- math: correctly rounded (mpmath) and platform libm (numpy) results.
tiny = [5e-324, 1e-310, 2.2250738585072014e-308, 1e-300, 1e-20, 1e-9, 1e-5]
exp_x = np.concatenate([rng.uniform(-745, 709.7, 800), rng.uniform(-1, 1, 200),
                        np.array(tiny), -np.array(tiny), [0.0, 709.78, -745.13, 0.5 * np.log(2)]])
ln_x = np.concatenate([10 ** rng.uniform(-307, 308, 600), rng.uniform(0.5, 2, 400),
                       1 + rng.uniform(-1e-6, 1e-6, 100), tiny, [1.0, 2.0, np.e, 1.7976931348623157e308]])
tanh_x = np.concatenate([rng.uniform(-25, 25, 800), rng.uniform(-1, 1, 200),
                         np.array(tiny), -np.array(tiny), [0.0, 0.55, 22.0, 23.0, 1e3, -1e3]])
lgamma_x = np.concatenate([rng.uniform(0.01, 30, 500), 10 ** rng.uniform(1.5, 8, 150),
                           np.arange(1, 60) / 2.0])
sincos_kn = [(k, n) for n in [1, 2, 3, 5, 7, 8, 12, 97, 150, 199, 256, 1000, 59049]
             for k in sorted(set(int(v) for v in np.linspace(0, n - 1, 25)))]
dump('math', {
    'exp': against(mp.exp, np.exp, exp_x),
    'ln': against(mp.log, np.log, ln_x),
    'tanh': against(mp.tanh, np.tanh, tanh_x),
    'lgamma': against(mp.loggamma, special.gammaln, lgamma_x),
    'sincos': {'k': [k for k, _ in sincos_kn], 'n': [n for _, n in sincos_kn],
               'sin': [float(mp.sin(2 * mp.pi * k / n)) for k, n in sincos_kn],
               'cos': [float(mp.cos(2 * mp.pi * k / n)) for k, n in sincos_kn]},
})

# --- stats: scipy is the reference tedana calls.
ab = [(0.5, 0.5), (1, 0.5), (0.5, 1), (2.5, 0.5), (74, 0.5), (5000, 0.5), (3, 7), (40, 60), (0.1, 30)]
inc = [(a, b, float(x)) for a, b in ab for x in np.linspace(0.001, 0.999, 23)]
dfs = [1, 2, 3, 5, 8, 13, 98, 148, 1000, 60000]
ts = [-40, -8, -3, -1.96, -0.5, -1e-3, 0, 1e-3, 0.5, 1.96, 3, 8, 40]
stdtr = [(d, float(t)) for d in dfs for t in ts]
fppf = [(d, q) for d in range(1, 11) for q in (0.95, 0.975, 0.99)]
dump('stats', {
    'betainc': {'a': [v[0] for v in inc], 'b': [v[1] for v in inc], 'x': [v[2] for v in inc],
                'y': [float(special.betainc(*v)) for v in inc]},
    'stdtr': {'df': [v[0] for v in stdtr], 't': [v[1] for v in stdtr],
              'p': [float(stats.t.cdf(t, df=d)) for d, t in stdtr]},
    'f_ppf': {'dfd': [v[0] for v in fppf], 'q': [v[1] for v in fppf],
              'f': [float(stats.f.ppf(q, 1, d)) for d, q in fppf]},
})

# --- linalg: LAPACK through scipy (gesdd, syevr) and numpy (gelsd).
def ill(m, n):
    u, _ = np.linalg.qr(rng.standard_normal((m, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return u @ np.diag(np.logspace(0, -8, n)) @ v.T


svd = []
for a in [rng.standard_normal((200, 7)), rng.standard_normal((9, 9)), ill(60, 6),
          rng.standard_normal((3, 11))]:
    u, s, vt = linalg.svd(a, full_matrices=False)
    svd.append({'a': mat(a), 'u': mat(u), 's': floats(s), 'vt': mat(vt)})
eig = []
for n in (1, 2, 9, 30):
    b = rng.standard_normal((n, n))
    w, v = linalg.eigh(b + b.T)
    eig.append({'a': mat(b + b.T), 'w': floats(w), 'v': mat(v)})
x, y = rng.standard_normal((100, 4)), rng.standard_normal((100, 3))
dump('linalg', {'svd': svd, 'eigh': eig,
                'lstsq': {'x': mat(x), 'y': mat(y), 'b': mat(np.linalg.lstsq(x, y, rcond=None)[0])}})

# --- fft: numpy's pocketfft, the backend scipy.fftpack/fftconvolve also use.
fft = []
for n in [1, 2, 3, 4, 5, 7, 8, 12, 16, 97, 128, 150, 199, 256, 299]:
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    f = np.fft.fft(z)
    fft.append({'re': floats(z.real), 'im': floats(z.imag), 'fre': floats(f.real), 'fim': floats(f.imag)})
v3 = rng.standard_normal((5, 6, 7))
f3 = np.fft.fftn(v3)
dump('fft', {'fft': fft, 'fftn': {'shape': list(v3.shape), 'x': floats(v3),
                                  'fre': floats(f3.real), 'fim': floats(f3.imag)}})

# --- rng: numpy legacy RandomState, as sklearn's FastICA w_init draws it.
rs = np.random.RandomState(42)
key = rs.get_state()[1]
dump('rng', {'seed': 42, 'key': [int(k) for k in key[:16]] + [int(key[-1])],
             'random_sample': floats(np.random.RandomState(42).random_sample(8)),
             'normal_15x15': floats(np.random.RandomState(42).normal(size=(15, 15))),
             'normal_13x13': floats(np.random.RandomState(42).normal(size=(13, 13)))})

# --- reductions and image ops tedana's masking depends on bit for bit.
from nilearn.image.image import smooth_array

sums = [rng.standard_normal(int(n)) * 10 ** rng.uniform(-3, 3, int(n)) for n in
        [1, 2, 7, 8, 9, 15, 16, 17, 100, 127, 128, 129, 255, 256, 300, 1000]]
logx = np.concatenate([np.arange(1, 70000, 61), rng.uniform(1, 1e5, 1000), 10 ** rng.uniform(-30, 30, 300)])
logx = logx.astype(np.float32)
vol = rng.standard_normal((7, 6, 5)) * 100 + 500
dump('image', {
    'sum64': [{'x': floats(a), 's': float(np.sum(a))} for a in sums],
    'sum32': [{'x': floats(a.astype(np.float32)), 's': float(np.sum(a.astype(np.float32)))} for a in sums],
    'log32': {'x': floats(logx), 'y': floats(np.log(logx))},
    'smooth': {'shape': list(vol.shape), 'x': floats(vol),
               'y': floats(smooth_array(vol.copy(), affine=np.eye(4), fwhm=1, ensure_finite=True, copy=False))},
})

print('numpy', np.__version__, 'scipy', scipy.__version__, 'mpmath', mp.__version__)

# --- numpy Generator (PCG64), as tedana's max_rp_corr draws its subsamples.
pcg = {}
for seed in (0, 42, 43, 2**40 + 7):
    g = np.random.default_rng(seed)
    st = g.bit_generator.state['state']
    pcg[str(seed)] = {'state': str(st['state']), 'inc': str(st['inc']),
                      'u64': [int(v) for v in np.random.default_rng(seed).integers(0, 2**64, 4, dtype=np.uint64, endpoint=False)],
                      'choice_100_90': [int(v) for v in np.random.default_rng(seed).choice(100, 90, replace=False)],
                      'choice_150_135x3': [int(v) for _ in range(1) for v in np.concatenate([g2.choice(150, 135, replace=False) for g2 in [np.random.default_rng(seed)] for _ in range(3)])]}
dump('pcg64', pcg)
