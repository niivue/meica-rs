//! tedana's default workflow, stage by stage.

use crate::io::Nifti;
use crate::linalg::Mat;
use crate::mapca::{Criterion, MaPca, mapca};
use crate::metrics::{Inputs, Metrics, generate_metrics};
use crate::selection::Table;
use crate::stats::mean_std;
use crate::{combine, decay, mask};

pub struct Combined {
    /// EPI mask (NIfTI order) and its voxels in numpy C order: the rows of every array below.
    pub epi_mask: Vec<bool>,
    pub voxels: Vec<usize>,
    /// voxel x echo x time, float32 as tedana loads it.
    pub data: Vec<f32>,
    pub masksum: Vec<i32>,
    pub t2s: Vec<f64>,
    pub s0: Vec<f64>,
    /// voxel x time.
    pub optcom: Vec<f64>,
}

/// tedana `check_te_values`: TEs all in seconds (< 1, BIDS) or all in ms (>= 1, divided by
/// 1000), ascending. Everything downstream works in seconds.
pub fn check_te_values(tes: &[f64]) -> Result<Vec<f64>, String> {
    if tes.is_empty() || tes.iter().any(|t| !t.is_finite()) {
        return Err("TE values must be finite numbers, one per echo".into());
    }
    let out = if tes.iter().all(|&t| t > 0.0 && t < 1.0) {
        tes.to_vec()
    } else if tes.iter().all(|&t| t >= 1.0) {
        tes.iter().map(|t| t / 1000.0).collect()
    } else {
        return Err("TE values must be positive and either all in seconds (< 1) or all in milliseconds (>= 1)".into());
    };
    if tes.windows(2).any(|w| w[1] <= w[0]) {
        return Err(
            "TE values must be in ascending order, matching the order of the input files".into(),
        );
    }
    Ok(out)
}

/// Mask (`user_mask`, tedana --mask, else nilearn's EPI mask of echo 1), adaptive mask,
/// log-linear T2*/S0 and optimal combination (TEs in seconds).
pub fn combine(
    echoes: &[Nifti],
    tes: &[f64],
    user_mask: Option<Vec<bool>>,
) -> Result<Combined, String> {
    if echoes.len() < 2 || echoes.len() != tes.len() {
        return Err(format!(
            "{} echo files but {} echo times; ME-ICA needs at least 2 echoes",
            echoes.len(),
            tes.len()
        ));
    }
    let dims = echoes[0].dims;
    if echoes.iter().any(|e| e.dims != dims) {
        return Err("every echo needs the same dimensions".into());
    }
    let (d, n_vol) = ([dims[0], dims[1], dims[2]], dims[3]);
    let nvox = d[0] * d[1] * d[2];
    if nvox < 2 {
        return Err("ME-ICA needs a 3D image of at least 2 voxels".into());
    }
    if user_mask.as_ref().is_some_and(|m| m.len() != nvox) {
        return Err("the mask does not match the image grid".into());
    }
    let epi_mask = user_mask.unwrap_or_else(|| mask::epi_mask(|i| echoes[0].value(i), d, n_vol));
    let voxels = mask::c_order(d, &epi_mask);
    if voxels.is_empty() {
        return Err("the brain mask is empty".into());
    }
    let mut data = Vec::with_capacity(voxels.len() * echoes.len() * n_vol);
    for &i in &voxels {
        for e in echoes {
            data.extend((0..n_vol).map(|t| e.value(i + nvox * t) as f32));
        }
    }
    let masksum = mask::adaptive_mask(&data, echoes.len(), n_vol);
    if masksum.iter().all(|&m| m == 0) {
        return Err(
            "no voxel has good signal in every volume (an all-zero volume or echo?)".into(),
        );
    }
    let (mut t2s, mut s0) = decay::fit_loglinear(&data, tes, &masksum, n_vol);
    decay::modify_maps(&mut t2s, &mut s0, tes, &masksum);
    let optcom = combine::optcom(&data, tes, &masksum, &t2s, n_vol);
    Ok(Combined {
        epi_mask,
        voxels,
        data,
        masksum,
        t2s,
        s0,
        optcom,
    })
}

/// tedpca for the maPCA criteria: the classification voxels (masksum >= 3) in mapca's NIfTI
/// order, maPCA on their optimal combination, and the ICA input `kept_data` (rows in that order,
/// each z-scored over time). Every component is kept, and the PCA metrics' sign flips cancel in
/// kept_data, so they are left to the metrics stage.
pub struct Reduced {
    /// NIfTI index of each kept_data row.
    pub positions: Vec<usize>,
    pub mapca: MaPca,
    pub kept_data: Mat,
}

pub fn tedpca(c: &Combined, dims: [usize; 3], criterion: Criterion) -> Result<Reduced, String> {
    let n_vol = c.optcom.len() / c.voxels.len();
    let mut row_of = vec![usize::MAX; dims[0] * dims[1] * dims[2]];
    for (v, &i) in c.voxels.iter().enumerate() {
        row_of[i] = v;
    }
    let rows: Vec<usize> = row_of
        .iter()
        .copied()
        .filter(|&v| v != usize::MAX && c.masksum[v] >= 3)
        .collect();
    if rows.is_empty() {
        return Err(
            "No voxels have good signal in 3+ echoes; ME-ICA needs at least 3 echoes".into(),
        );
    }
    let positions: Vec<usize> = rows.iter().map(|&v| c.voxels[v]).collect();
    let mut x = Mat::zeros(rows.len(), n_vol);
    for (r, &v) in rows.iter().enumerate() {
        for t in 0..n_vol {
            x[(r, t)] = c.optcom[v * n_vol + t];
        }
    }
    let m = mapca(&x, dims, &positions, criterion)?;
    // (u * varex) components, then scipy zscore over time per row.
    let mut weighted = m.u.clone();
    for (k, &ev) in m.explained_variance.iter().enumerate() {
        weighted.col_mut(k).iter_mut().for_each(|v| *v *= ev);
    }
    let mut kept = crate::linalg::matmul(&weighted, &m.components);
    let mut row = vec![0.0; n_vol];
    for r in 0..kept.rows {
        for (t, x) in row.iter_mut().enumerate() {
            *x = kept[(r, t)];
        }
        let (mean, std) = mean_std(&row);
        for t in 0..n_vol {
            kept[(r, t)] = (row[t] - mean) / std;
        }
    }
    Ok(Reduced {
        positions,
        mapca: m,
        kept_data: kept,
    })
}

/// Component metrics on the classification voxels (masksum >= 3, tedana's C order) and the
/// sign-flipped mixing matrix.
pub fn component_metrics(
    c: &Combined,
    dims: [usize; 3],
    tes: &[f64],
    mixing: &Mat,
) -> (Metrics, Mat) {
    let (n_echo, n_vol) = (tes.len(), mixing.rows);
    let rows: Vec<usize> = (0..c.voxels.len()).filter(|&v| c.masksum[v] >= 3).collect();
    let mut data = Vec::with_capacity(rows.len() * n_echo * n_vol);
    let mut optcom = Vec::with_capacity(rows.len() * n_vol);
    for &v in &rows {
        data.extend_from_slice(&c.data[v * n_echo * n_vol..(v + 1) * n_echo * n_vol]);
        optcom.extend_from_slice(&c.optcom[v * n_vol..(v + 1) * n_vol]);
    }
    let masksum: Vec<i32> = rows.iter().map(|&v| c.masksum[v]).collect();
    let positions: Vec<usize> = rows.iter().map(|&v| c.voxels[v]).collect();
    let inputs = Inputs {
        data: &data,
        optcom: &optcom,
        masksum: &masksum,
        tes,
        dims,
        positions: &positions,
    };
    generate_metrics(&inputs, mixing)
}

/// tedana's metric columns, in its preferred order.
pub fn metric_columns(m: &Metrics) -> Vec<(&'static str, Vec<f64>)> {
    let f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    vec![
        ("kappa", m.kappa.clone()),
        ("rho", m.rho.clone()),
        ("variance explained", m.variance_explained.clone()),
        (
            "normalized variance explained",
            m.normalized_variance_explained.clone(),
        ),
        ("countsigFT2", f(&m.countsig_ft2)),
        ("countsigFS0", f(&m.countsig_fs0)),
        ("dice_FT2", m.dice_ft2.clone()),
        ("dice_FS0", m.dice_fs0.clone()),
        ("countnoise", f(&m.countnoise)),
        ("signal-noise_t", m.signal_noise_t.clone()),
        ("signal-noise_p", m.signal_noise_p.clone()),
        ("d_table_score", m.d_table_score.clone()),
        (
            "optimal sign",
            m.optimal_sign.iter().map(|&s| s as f64).collect(),
        ),
    ]
}

/// The component table tedana selects on.
pub fn component_table(m: &Metrics) -> Table {
    Table::new(
        metric_columns(m)
            .into_iter()
            .map(|(n, v)| (n.into(), v))
            .collect(),
    )
}
