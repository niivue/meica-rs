#!/usr/bin/env python3
"""Write intermediates tedana computes but does not save, using tedana's and mapca's own code.

Usage: gen_stage_reference.py EXAMPLE_DIR OUT_DIR
  epi_mask.nii.gz   nilearn's EPI mask of echo 1
  kept_data.nii.gz  tedpca's ICA input (f64, rows placed in mapca's F voxel order)
  mapca.json        maPCA's internal decisions and spectra
  fastica.json      FastICA on kept_data with tedana's defaults (seed 42, 500 iterations)
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from mapca import MovingAveragePCA
from nilearn import masking
from nilearn.masking import compute_epi_mask
from scipy import stats
from sklearn.decomposition import FastICA

from tedana import combine, decay, io, utils
from tedana.decomposition.ica import f_ica

example, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
echoes = [str(example / f'sub-01_task-motor_echo-{e}_bold.nii.gz') for e in (1, 2, 3)]
tes = utils.check_te_values([15.2, 35.32, 55.42])

# tedana_workflow up to tedpca, with its defaults.
ref_img = io.load_ref_img(data=echoes, n_echos=3)
mask_img = compute_epi_mask(ref_img)
nib.save(nib.Nifti1Image(np.asarray(mask_img.dataobj).astype(np.uint8), mask_img.affine), out / 'epi_mask.nii.gz')
data_cat = io.load_data_nilearn(echoes, mask_img=mask_img, n_echos=3)
mask_denoise, masksum = utils.make_adaptive_mask(data_cat, threshold=1, methods=['dropout'])
masksum_clf = masksum.copy()
masksum_clf[masksum_clf < 3] = 0
mask_denoise, mask_clf = mask_denoise.astype(bool), masksum_clf.astype(bool)
t2s, s0, *_ = decay.fit_decay(data_cat[mask_denoise], tes, masksum[mask_denoise], 'loglin')
t2s, s0, _, _ = decay.modify_t2s_s0_maps(t2s, s0, masksum[mask_denoise], tes)
data_optcom = combine.make_optcom(data_cat, tes, masksum, t2s=utils.unmask(t2s, mask_denoise))

# decomposition.tedpca for algorithm="aic", minus the PCA metrics (their sign flips cancel in kept_data).
data = data_optcom[mask_clf, :]
data_img = masking.unmask(utils.unmask(data, mask_clf).T, mask_img)
clf_img = masking.unmask(mask_clf, mask_img)
ma_pca = MovingAveragePCA(criterion='aic', normalize=True)
ma_pca.fit_transform(data_img, clf_img)
varex, comp_ts = ma_pca.explained_variance_, ma_pca.components_.T
kept = np.dot(ma_pca.u_ * varex[None, :], comp_ts.T)
kept = stats.zscore(kept, axis=1)

clf = np.asarray(clf_img.dataobj) > 0
vol = np.zeros(clf.shape + (kept.shape[1],))
flat = vol.reshape(-1, kept.shape[1], order='F')
flat[clf.reshape(-1, order='F')] = kept
vol = flat.reshape(vol.shape, order='F')
nib.save(nib.Nifti1Image(vol, mask_img.affine), out / 'kept_data.nii.gz')

sub = ma_pca.subsampling_
(out / 'mapca.json').write_text(json.dumps({
    'n_samples': int(sub['total_num_samples']),
    'iid_subsample_input': [float(v) for v in sub['IID_subsample_input']],
    'subsample_depth': int(sub['used_IID_subsample_depth']),
    'effective_samples': float(sub['effective_num_IID_samples']),
    'aic': {'n': int(ma_pca.aic_['n_components']), 'curve': [float(v) for v in ma_pca.aic_['value']]},
    'kic': {'n': int(ma_pca.kic_['n_components']), 'curve': [float(v) for v in ma_pca.kic_['value']]},
    'mdl': {'n': int(ma_pca.mdl_['n_components']), 'curve': [float(v) for v in ma_pca.mdl_['value']]},
    'varex_90': int(ma_pca.varexp_90_['n_components']),
    'varex_95': int(ma_pca.varexp_95_['n_components']),
    'cumulative_varex': [float(v) for v in ma_pca.all_['explained_variance_total']],
    'explained_variance': [float(v) for v in varex],
    'components': [[float(v) for v in row] for row in ma_pca.components_],
}) + '\n')

# decomposition.tedica -> f_ica: tedana's z-scored mixing, plus sklearn's raw attributes for seed 42.
n_comp = int(ma_pca.aic_['n_components'])
mixing, seed = f_ica(kept, n_components=n_comp, fixed_seed=42, maxit=500, maxrestart=10)
ica = FastICA(n_components=n_comp, algorithm='parallel', fun='logcosh', max_iter=500, random_state=42)
ica.fit(kept)
(out / 'fastica.json').write_text(json.dumps({
    'seed': int(seed),
    'n_iter': int(ica.n_iter_),
    'whitening': [[float(v) for v in row] for row in ica.whitening_],
    'components': [[float(v) for v in row] for row in ica.components_],
    'mixing_raw': [[float(v) for v in row] for row in ica.mixing_],
    'mixing': [[float(v) for v in row] for row in mixing],
}) + '\n')
