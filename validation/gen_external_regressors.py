#!/usr/bin/env python3
"""Regressors named as tedana's demo external-regressor trees expect, for parity tests.

Usage: gen_external_regressors.py EXAMPLE_DIR OUT_TSV
  Mot_X..Mot_Yaw   trans_x/y/z (mm) and rot_x/y/z (rad) from the example's confounds TSV
  csf              echo-3 mean of the brightest 2% of its temporal-mean EPI-mask voxels
  signal_left/right  events boxcars convolved with SPM's canonical HRF
The demo trees match '^mot_.*$' case-insensitively, so one file serves all three.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import compute_epi_mask
from scipy.stats import gamma

example, out = Path(sys.argv[1]), Path(sys.argv[2])
conf = pd.read_table(example / 'sub-01_task-motor_desc-confounds_timeseries.tsv')
names = {'trans_x': 'Mot_X', 'trans_y': 'Mot_Y', 'trans_z': 'Mot_Z',
         'rot_x': 'Mot_Pitch', 'rot_y': 'Mot_Roll', 'rot_z': 'Mot_Yaw'}
reg = conf[list(names)].rename(columns=names)

e1 = nib.load(example / 'sub-01_task-motor_echo-1_bold.nii.gz')
e3 = np.asarray(nib.load(example / 'sub-01_task-motor_echo-3_bold.nii.gz').dataobj, dtype=float)
mask = np.asarray(compute_epi_mask(e1).dataobj) > 0
mean = e3.mean(axis=3)
reg['csf'] = e3[mask & (mean >= np.percentile(mean[mask], 98))].mean(axis=0)

tr, n = float(e1.header.get_zooms()[3]), e3.shape[3]
t = np.arange(0, 32, tr)
hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
events = pd.read_table(example / 'sub-01_task-motor_events.tsv')
for kind in ('left', 'right'):
    box = np.zeros(n)
    for _, e in events[events.trial_type == kind].iterrows():
        box[int(e.onset / tr):int((e.onset + e.duration) / tr)] = 1
    reg[f'signal_{kind}'] = np.convolve(box, hrf)[:n]
reg.to_csv(out, sep='\t', index=False)
