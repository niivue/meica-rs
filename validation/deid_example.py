#!/usr/bin/env python3
"""Scrub, crop and trim the Moto 3-echo run into a small int16 example.

Usage: deid_example.py IN_DIR OUT_DIR STEM [--vols 100] [--margin 2]
STEM is the source name before `_echo-N` (the raw subject ID stays out of this repo).
"""
import argparse
import csv
import gzip
import struct
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.masking import compute_epi_mask

DST = 'sub-01_task-motor'
KEEP = ['EchoTime', 'EchoNumber', 'RepetitionTime', 'SliceTiming', 'PhaseEncodingDirection',
        'TotalReadoutTime', 'EffectiveEchoSpacing', 'MultibandAccelerationFactor', 'FlipAngle',
        'MagneticFieldStrength', 'Manufacturer', 'ManufacturersModelName', 'TaskName']

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('src', type=Path)
ap.add_argument('dst', type=Path)
ap.add_argument('stem')
ap.add_argument('--vols', type=int, default=100)
ap.add_argument('--margin', type=int, default=2)
a = ap.parse_args()
a.dst.mkdir(parents=True, exist_ok=True)

imgs = [nib.load(a.src / f'{a.stem}_echo-{e}_part-mag_bold.nii.gz') for e in (1, 2, 3)]
data = [np.asarray(i.dataobj)[..., :a.vols] for i in imgs]

# Brain box from the mask tedana itself starts from (nilearn's EPI mask of echo 1).
mask = np.asarray(compute_epi_mask(imgs[0].slicer[..., :a.vols]).dataobj) > 0
box = tuple(slice(max(ix.min() - a.margin, 0), min(ix.max() + a.margin + 1, n))
            for ix, n in zip(np.where(mask), mask.shape))

# One shared slope keeps echoes on the same scale; ceil(v/2) fits uint16 into int16. Float input
# (motion-corrected) is clipped at 0: magnitude data, the negatives are spline ringing.
slope = int(np.ceil(max(float(d.max()) for d in data) / 32767))
for e, (img, d) in enumerate(zip(imgs, data), start=1):
    raw = np.ceil(np.clip(d[box], 0, None) / slope)
    out = nib.Nifti1Image(raw.astype(np.int16), img.slicer[box].affine)
    hdr = out.header
    hdr.set_xyzt_units('mm', 'sec')
    hdr['pixdim'][4] = img.header['pixdim'][4]
    out.set_qform(img.slicer[box].affine, int(img.header['qform_code']))
    out.set_sform(img.slicer[box].affine, int(img.header['sform_code']))
    # nibabel drops a slope it did not choose, so patch scl_slope/scl_inter (offset 112) after
    # serializing; mtime=0 keeps the gzip bytes reproducible.
    blob = bytearray(out.to_bytes())
    blob[112:120] = struct.pack('<ff', slope, 0)
    path = a.dst / f'{DST}_echo-{e}_bold.nii.gz'
    path.write_bytes(gzip.compress(bytes(blob), compresslevel=9, mtime=0))
    meta = json.loads((a.src / f'{a.stem}_echo-{e}_part-mag_bold.json').read_text())
    (a.dst / f'{DST}_echo-{e}_bold.json').write_text(
        json.dumps({k: meta[k] for k in KEEP if k in meta}, indent=2) + '\n')

tr = float(imgs[0].header['pixdim'][4])
with open(a.src / f'{a.stem}_events.tsv') as f:
    rows = [r for r in csv.DictReader(f, delimiter='\t')
            if float(r['onset']) + float(r['duration']) <= a.vols * tr]
with open(a.dst / f'{DST}_events.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['onset', 'duration', 'trial_type'], delimiter='\t',
                       lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
print('box', [(s.start, s.stop) for s in box], 'shape', data[0][box].shape, 'slope', slope)
