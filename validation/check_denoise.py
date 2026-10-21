"""meica --denoise on a run's own optcom (with N dummy volumes prepended) must reproduce the run's
desc-denoised_bold to 1e-6 of its range, pass the dummies through, leave the input unchanged when
nothing is rejected, remove every mixing column when everything is, and refuse a length mismatch.
usage: check_denoise.py MEICA RUN_DIR WORK_DIR"""
import subprocess
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

meica, run, work, n = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), 2
work.mkdir(parents=True, exist_ok=True)
opt = nb.load(run / 'desc-optcom_bold.nii.gz')
ref = nb.load(run / 'desc-denoised_bold.nii.gz').get_fdata()
x = opt.get_fdata(dtype=np.float32)
bold = np.concatenate([np.random.default_rng(0).normal(size=x.shape[:3] + (n,)).astype(np.float32), x], 3)
nb.save(nb.Nifti1Image(bold, opt.affine, opt.header), work / 'bold.nii.gz')
metrics = (run / 'desc-tedana_metrics.tsv').read_text()


def denoise(metrics_text, dummies=n):
    (work / 'metrics.tsv').write_text(metrics_text)
    out = work / 'out.nii.gz'
    out.unlink(missing_ok=True)
    p = subprocess.run([meica, '--denoise', work / 'bold.nii.gz', '--mixing', run / 'desc-ICA_mixing.tsv',
                        '--metrics', work / 'metrics.tsv', '-o', out, '--dummy-scans', str(dummies)])
    return nb.load(out).get_fdata() if p.returncode == 0 else None


def reclassify(label):
    head, *rows = metrics.splitlines()
    c = head.split('\t').index('classification')
    return '\n'.join([head] + ['\t'.join(v if i != c else label for i, v in enumerate(r.split('\t'))) for r in rows])


y = denoise(metrics)
assert 'rejected' in metrics, 'the run must reject something'
assert np.array_equal(y[..., :n], bold[..., :n]), 'dummy volumes changed'
mask = ref.any(3)
err = np.abs(y[..., n:] - ref)[mask].max() / np.ptp(ref[mask])
assert err < 1e-6, f'denoised differs by {err:.2e} of range'
assert np.array_equal(denoise(reclassify('accepted')), bold), 'all accepted must return the input'
mix = np.loadtxt(run / 'desc-ICA_mixing.tsv', skiprows=1, ndmin=2)
r = denoise(reclassify('rejected'))[mask][:, n:]
r -= r.mean(1, keepdims=True)
assert np.abs(r @ mix).max() / np.abs(r).sum(1).max() < 1e-4, 'all rejected must leave no mixing column'
assert denoise(metrics, dummies=n + 1) is None, 'a length mismatch must fail'
print(f'check_denoise: ok ({err:.1e} of range)')
