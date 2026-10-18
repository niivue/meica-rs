#!/usr/bin/env python3
"""compare_outputs.py must fail when one side has a NaN where the other has a value."""
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

with tempfile.TemporaryDirectory() as d:
    ref, out = Path(d, 'ref'), Path(d, 'out')
    for p, v in [(ref, [1, 2]), (out, [1, np.nan])]:
        p.mkdir()
        nib.save(nib.Nifti1Image(np.array(v, np.float32).reshape(2, 1, 1), np.eye(4)), p / 'x.nii.gz')
    run = subprocess.run([sys.executable, Path(__file__).with_name('compare_outputs.py'), ref, out],
                         capture_output=True, text=True)
    assert run.returncode == 1, run.stdout
print('compare_outputs NaN check ok')
