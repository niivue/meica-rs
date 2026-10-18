#!/usr/bin/env python3
"""Compare a meica output directory with tedana's, file by file.

Usage: compare_outputs.py TEDANA_DIR MEICA_DIR [--tol 1e-6]
Images, TSV columns and JSON numbers must agree to a relative tolerance (scaled by each file's
largest finite magnitude), NaN and infinity exactly; text and integers must be equal; lists of names compare as sets. Exits 1 on
any difference beyond the tolerance or a missing file.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SKIP = {'dataset_description.json', 'desc-tedana_registry.json', 'report.txt', 'references.bib',
        'tedana_report.html', 'open_rica_report.py', 'tedana_call.sh'}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('ref', type=Path)
ap.add_argument('out', type=Path)
ap.add_argument('--tol', type=float, default=1e-6)  # measured worst: 1.5e-8 example, 1e-7 full scan
a = ap.parse_args()
failures = []


def report(name, worst, ok, note=''):
    print(f'{"ok  " if ok else "FAIL"} {name}: worst {worst:.3g}{note}')
    if not ok:
        failures.append(name)


def numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def same_nonfinite(r, o):
    """NaN and infinity must sit at the same places with the same values; only finite values get a tolerance."""
    m = ~np.isfinite(r)
    return np.array_equal(m, ~np.isfinite(o)) and np.array_equal(r[m], o[m], equal_nan=True)


def walk(r, o, path, errs):
    """Collect (path, relative error) for numbers; differences for everything else."""
    if isinstance(r, dict) and isinstance(o, dict):
        for k in set(r) | set(o):
            if k not in r or k not in o:
                errs.append((f'{path}/{k}', 'missing in ' + ('ours' if k in r else 'tedana')))
            else:
                walk(r[k], o[k], f'{path}/{k}', errs)
    elif isinstance(r, list) and isinstance(o, list):
        if all(isinstance(v, str) for v in r + o):
            if sorted(r) != sorted(o):
                errs.append((path, f'{r} != {o}'))
        elif len(r) != len(o):
            errs.append((path, f'length {len(r)} != {len(o)}'))
        else:
            for i, (x, y) in enumerate(zip(r, o)):
                walk(x, y, f'{path}[{i}]', errs)
    elif numeric(r) and numeric(o):
        if not same_nonfinite(np.array([r], float), np.array([o], float)):
            errs.append((path, f'{r!r} != {o!r}'))
        elif np.isfinite(r):
            errs.append((path, abs(r - o) / max(abs(r), 1e-300)))
    elif r != o:
        errs.append((path, f'{r!r} != {o!r}'))


for ref in sorted(a.ref.iterdir()):
    name = ref.name
    if name in SKIP or ref.is_dir() or name.startswith('tedana_20'):
        continue
    out = a.out / name
    if not out.exists():
        report(name, float('nan'), False, ' (missing)')
        continue
    if name.endswith('.nii.gz'):
        r, o = nib.load(ref), nib.load(out)
        rd, od = np.asanyarray(r.dataobj), np.asanyarray(o.dataobj)
        same_geometry = r.shape == o.shape and np.allclose(r.affine, o.affine) and rd.dtype == od.dtype
        if not same_geometry:
            report(name, float('nan'), False, ' (shape, affine or dtype differs)')
            continue
        rf, of = rd.astype(float), od.astype(float)
        if not same_nonfinite(rf, of):
            report(name, float('inf'), False, ' (NaN/inf pattern differs)')
            continue
        finite = np.isfinite(rf)
        scale = np.max(np.abs(rf[finite]), initial=0.0) or 1.0
        worst = float(np.max(np.abs(rf - of)[finite], initial=0.0)) / scale
        exact = float(np.mean((rd == od) | ~finite))
        report(name, worst, worst <= a.tol, f' of max, {exact:.1%} bit-identical')
    elif name.endswith('.tsv'):
        r = [line.split('\t') for line in ref.read_text().splitlines()]
        o = [line.split('\t') for line in out.read_text().splitlines()]
        if r[0] != o[0] or len(r) != len(o):
            report(name, float('nan'), False, f' (header or rows differ: {r[0]} vs {o[0]})')
            continue
        worst, note = 0.0, ''
        for j, col in enumerate(r[0]):
            rv, ov = [row[j] for row in r[1:]], [row[j] for row in o[1:]]
            try:
                rf = np.array([np.nan if v == 'n/a' else float(v) for v in rv])
                of = np.array([np.nan if v == 'n/a' else float(v) for v in ov])
            except ValueError:
                same = all(sorted(x.split(',')) == sorted(y.split(',')) for x, y in zip(rv, ov))
                if not same:
                    note += f' {col} differs'
                    worst = float('inf')
                continue
            if not same_nonfinite(rf, of):
                note += f' {col} n/a pattern differs'
                worst = float('inf')
                continue
            finite = np.isfinite(rf)
            scale = np.max(np.abs(rf[finite]), initial=0.0) or 1.0
            worst = max(worst, float(np.max(np.abs(rf - of)[finite], initial=0.0)) / scale)
        report(name, worst, worst <= a.tol, note)
    elif name.endswith('.json'):
        errs = []
        walk(json.loads(ref.read_text()), json.loads(out.read_text()), '', errs)
        nums = [e for p, e in errs if isinstance(e, float)]
        other = [(p, e) for p, e in errs if not isinstance(e, float)]
        worst = max(nums, default=0.0)
        report(name, worst, worst <= a.tol and not other, ''.join(f'\n       {p}: {e}' for p, e in other[:8]))

print(f'{len(failures)} file(s) differ' if failures else 'all files agree')
sys.exit(1 if failures else 0)
