//! tedana `metrics.collect.generate_metrics` for the metrics every bundled tree uses: kappa, rho,
//! variance explained (raw and normalized), countsigFT2/FS0, dice_FT2/FS0, countnoise,
//! signal-noise_t/p and d_table_score.
//!
//! Rows are the classification voxels in tedana's C order; maps are voxels x components with
//! each component's map contiguous.

use crate::linalg::{Mat, Qr};
use crate::mask::{Dims, neighbours};
use crate::math::ln;
use crate::selection::decision_table_score;
use crate::stats::{
    demean, f_ppf_dfn1, percentile_higher, rankdata, sum, welch_ttest, zscore, zscore_columns,
};

pub struct Inputs<'a> {
    /// voxel x echo x time, float32 as tedana loads it.
    pub data: &'a [f32],
    /// voxel x time.
    pub optcom: &'a [f64],
    pub masksum: &'a [i32],
    pub tes: &'a [f64],
    /// Grid of the volume and each row's NIfTI index in it (for clustering).
    pub dims: Dims,
    pub positions: &'a [usize],
}

pub struct Metrics {
    pub kappa: Vec<f64>,
    pub rho: Vec<f64>,
    pub variance_explained: Vec<f64>,
    pub normalized_variance_explained: Vec<f64>,
    pub countsig_ft2: Vec<usize>,
    pub countsig_fs0: Vec<usize>,
    pub dice_ft2: Vec<f64>,
    pub dice_fs0: Vec<f64>,
    pub countnoise: Vec<usize>,
    pub signal_noise_t: Vec<f64>,
    pub signal_noise_p: Vec<f64>,
    pub d_table_score: Vec<f64>,
    pub optimal_sign: Vec<i32>,
}

/// tedana `get_coeffs`: least-squares betas of every voxel series on the columns of x.
pub(crate) fn coeffs(
    series: impl Fn(usize, &mut Vec<f64>),
    n_vox: usize,
    x: &Mat,
    add_const: bool,
) -> Mat {
    let mut design = x.clone();
    if add_const {
        design = Mat::zeros(x.rows, x.cols + 1);
        design.data[..x.data.len()].copy_from_slice(&x.data);
        design.col_mut(x.cols).fill(1.0);
    }
    let qr = Qr::new(&design);
    let mut out = Mat::zeros(n_vox, x.cols);
    let mut y = Vec::new();
    for v in 0..n_vox {
        series(v, &mut y);
        let b = qr.solve(&mut y);
        for c in 0..x.cols {
            out[(v, c)] = b[c];
        }
    }
    out
}

/// Betas of each row's series (row-major, `n_vol` long) on `x`.
pub(crate) fn row_coeffs(
    rows: &[f64],
    n_vol: usize,
    x: &Mat,
    transform: impl Fn(&mut Vec<f64>),
) -> Mat {
    coeffs(
        |v, y| {
            y.clear();
            y.extend_from_slice(&rows[v * n_vol..(v + 1) * n_vol]);
            transform(y);
        },
        rows.len() / n_vol,
        x,
        false,
    )
}

/// Voxels in 6-connected clusters of at least `csize` among `on` (grid order).
fn clusters(d: Dims, on: &[bool], csize: usize) -> Vec<bool> {
    let mut seen = vec![false; on.len()];
    let mut keep = vec![false; on.len()];
    let (mut stack, mut members) = (Vec::new(), Vec::new());
    for seed in 0..on.len() {
        if !on[seed] || seen[seed] {
            continue;
        }
        seen[seed] = true;
        stack.push(seed);
        members.clear();
        while let Some(i) = stack.pop() {
            members.push(i);
            for j in neighbours(d, i) {
                if on[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if members.len() >= csize {
            members.iter().for_each(|&i| keep[i] = true);
        }
    }
    keep
}

/// tedana `utils.threshold_map(sided="bi", binarize=True)` of one map: clusters of values
/// >= threshold and clusters of values <= -threshold. Voxels outside the rows are 0.
fn threshold_map(values: &[f64], inp: &Inputs, threshold: f64, csize: usize) -> Vec<bool> {
    let n = inp.dims.iter().product();
    let mut grid = vec![0.0; n];
    for (r, &i) in inp.positions.iter().enumerate() {
        grid[i] = values[r];
    }
    let pos = clusters(
        inp.dims,
        &grid.iter().map(|&v| v >= threshold).collect::<Vec<_>>(),
        csize,
    );
    let neg = clusters(
        inp.dims,
        &grid.iter().map(|&v| v <= -threshold).collect::<Vec<_>>(),
        csize,
    );
    inp.positions.iter().map(|&i| pos[i] || neg[i]).collect()
}

/// tedana `threshold_to_match`: lower a rank threshold on |betas| until clustering finds at
/// least `target` voxels, then step back once.
fn threshold_to_match(betas: &[f64], target: usize, inp: &Inputs, csize: usize) -> Vec<bool> {
    if target == 0 {
        return vec![false; betas.len()];
    }
    let ranks = rankdata(&betas.iter().map(|b| b.abs()).collect::<Vec<_>>());
    let step = (target / 10).max(1) as f64;
    let mut rank_thresh = betas.len() as f64 - target as f64;
    loop {
        let clmap = threshold_map(&ranks, inp, rank_thresh, csize);
        let found = clmap.iter().filter(|&&b| b).count();
        if rank_thresh <= 0.0 {
            return clmap;
        }
        if found > target || found == 0 {
            return threshold_map(&ranks, inp, rank_thresh + step, csize);
        }
        rank_thresh -= step;
    }
}

fn dice(a: &[bool], b: &[bool]) -> f64 {
    let total = a.iter().filter(|&&x| x).count() + b.iter().filter(|&&x| x).count();
    let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    if total == 0 {
        0.0
    } else {
        2.0 * inter as f64 / total as f64
    }
}

fn log10(x: f64) -> f64 {
    ln(x) / ln(10.0) // ponytail: numpy calls libm log10; this can differ by an ulp
}

/// Metrics and the sign-flipped mixing matrix (time x components).
pub fn generate_metrics(inp: &Inputs, mixing: &Mat) -> (Metrics, Mat) {
    let (n_vox, n_comp, n_echo) = (inp.masksum.len(), mixing.cols, inp.tes.len());
    let n_vol = mixing.rows;
    let csize = (((n_vox as f64 * 0.0005) as usize) + 5).max(20);
    let f_thresh = f_ppf_dfn1(1.0 - 0.05, (n_echo - 1) as f64);

    // map weight: betas of z-scored data on the z-scored mixing; skewness picks each sign.
    let mix_z = zscore_columns(mixing);
    let mut weights = row_coeffs(inp.optcom, n_vol, &mix_z, |y| zscore(y));
    let mut mixing = mixing.clone();
    let mut signs = vec![1; n_comp];
    for (c, sign) in signs.iter_mut().enumerate() {
        let col = weights.col(c);
        let mean = sum(col) / n_vox as f64;
        let m2 = sum(&col
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .collect::<Vec<_>>())
            / n_vox as f64;
        let m3 = sum(&col
            .iter()
            .map(|v| (v - mean) * (v - mean) * (v - mean))
            .collect::<Vec<_>>())
            / n_vox as f64;
        let degenerate = m2 <= (f64::EPSILON * mean) * (f64::EPSILON * mean);
        if !degenerate && m3 < 0.0 {
            *sign = -1;
            weights.col_mut(c).iter_mut().for_each(|v| *v = -*v);
            mixing.col_mut(c).iter_mut().for_each(|v| *v = -*v);
        }
    }

    // map optcom betas: on the demeaned optimal combination.
    let betas = row_coeffs(inp.optcom, n_vol, &mixing, |y| demean(y));

    // map FT2 / FS0: per echo betas (with a constant) fitted against the S0 and T2 models.
    let echo_betas: Vec<Mat> = (0..n_echo)
        .map(|e| {
            coeffs(
                |v, y| {
                    y.clear();
                    y.extend(
                        inp.data[(v * n_echo + e) * n_vol..(v * n_echo + e + 1) * n_vol]
                            .iter()
                            .map(|&x| x as f64),
                    );
                },
                n_vox,
                &mixing,
                true,
            )
        })
        .collect();
    let mu: Vec<f64> = (0..n_vox * n_echo)
        .map(|k| {
            let s: Vec<f64> = inp.data[k * n_vol..(k + 1) * n_vol]
                .iter()
                .map(|&x| x as f64)
                .collect();
            sum(&s) / n_vol as f64
        })
        .collect();
    let (mut f_t2, mut f_s0) = (Mat::zeros(n_vox, n_comp), Mat::zeros(n_vox, n_comp));
    for c in 0..n_comp {
        for v in 0..n_vox {
            let j = inp.masksum[v] as usize;
            if j < 3 {
                continue;
            }
            let b: Vec<f64> = (0..j).map(|e| echo_betas[e][(v, c)]).collect();
            let alpha = b.iter().fold(-0.0, |a, x| a + x.abs() * x.abs());
            let fstat = |x: &[f64]| {
                let num = (0..j).fold(-0.0, |a, e| a + b[e] * x[e]);
                let den = (0..j).fold(-0.0, |a, e| a + x[e] * x[e]);
                let coef = num / den;
                let sse = (0..j).fold(-0.0, |a, e| a + (b[e] - x[e] * coef) * (b[e] - x[e] * coef));
                let f = (alpha - sse) * (j - 1) as f64 / sse;
                if f > 500.0 { 500.0 } else { f }
            };
            let x1: Vec<f64> = (0..j).map(|e| mu[v * n_echo + e]).collect();
            let x2: Vec<f64> = (0..j).map(|e| inp.tes[e] * mu[v * n_echo + e]).collect();
            f_s0[(v, c)] = fstat(&x1);
            f_t2[(v, c)] = fstat(&x2);
        }
    }

    let mut m = Metrics {
        kappa: vec![0.0; n_comp],
        rho: vec![0.0; n_comp],
        variance_explained: Vec::new(),
        normalized_variance_explained: Vec::new(),
        countsig_ft2: vec![0; n_comp],
        countsig_fs0: vec![0; n_comp],
        dice_ft2: vec![0.0; n_comp],
        dice_fs0: vec![0.0; n_comp],
        countnoise: vec![0; n_comp],
        signal_noise_t: vec![0.0; n_comp],
        signal_noise_p: vec![0.0; n_comp],
        d_table_score: Vec::new(),
        optimal_sign: signs,
    };
    for c in 0..n_comp {
        let (z, ft2, fs0) = (weights.col(c), f_t2.col(c), f_s0.col(c));
        let w2: Vec<f64> = z.iter().map(|v| v * v).collect();
        let total = sum(&w2);
        m.kappa[c] = sum(&ft2.iter().zip(&w2).map(|(f, w)| f * w).collect::<Vec<_>>()) / total;
        m.rho[c] = sum(&fs0.iter().zip(&w2).map(|(f, w)| f * w).collect::<Vec<_>>()) / total;

        let z_thresh = percentile_higher(&z.iter().map(|v| v.abs()).collect::<Vec<_>>(), 95.0);
        let z_cl = threshold_map(z, inp, z_thresh, csize);
        let ft2_cl = threshold_map(ft2, inp, f_thresh, csize);
        let fs0_cl = threshold_map(fs0, inp, f_thresh, csize);
        m.countsig_ft2[c] = ft2_cl.iter().filter(|&&b| b).count();
        m.countsig_fs0[c] = fs0_cl.iter().filter(|&&b| b).count();
        let beta_t2_cl = threshold_to_match(betas.col(c), m.countsig_ft2[c], inp, csize);
        let beta_s0_cl = threshold_to_match(betas.col(c), m.countsig_fs0[c], inp, csize);
        m.dice_ft2[c] = dice(&beta_t2_cl, &ft2_cl);
        m.dice_fs0[c] = dice(&beta_s0_cl, &fs0_cl);

        let noise: Vec<bool> = z
            .iter()
            .zip(&z_cl)
            .map(|(v, &cl)| v.abs() > z_thresh && !cl)
            .collect();
        m.countnoise[c] = noise.iter().filter(|&&b| b).count();
        let unique_log = |pick: &dyn Fn(usize) -> bool| {
            let mut vals: Vec<f64> = (0..n_vox).filter(|&v| pick(v)).map(|v| ft2[v]).collect();
            vals.sort_by(f64::total_cmp);
            vals.dedup();
            vals.into_iter().map(log10).collect::<Vec<_>>()
        };
        let (t, p) = welch_ttest(&unique_log(&|v| z_cl[v]), &unique_log(&|v| noise[v]));
        m.signal_noise_t[c] = if t.is_nan() { 0.0 } else { t };
        m.signal_noise_p[c] = if p.is_nan() { 0.0 } else { p };
    }
    // variance explained: column sums of squared maps (sequential), as percentages.
    let varex = |maps: &Mat| {
        let compvar: Vec<f64> = (0..n_comp)
            .map(|c| maps.col(c).iter().fold(-0.0, |a, v| a + v * v))
            .collect();
        let total = sum(&compvar);
        compvar
            .iter()
            .map(|v| 100.0 * (v / total))
            .collect::<Vec<f64>>()
    };
    m.variance_explained = varex(&betas);
    m.normalized_variance_explained = varex(&weights);
    let as_f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    m.d_table_score = decision_table_score(
        &[&as_f(&m.countnoise)],
        &[
            &m.kappa,
            &m.dice_ft2,
            &m.signal_noise_t,
            &as_f(&m.countsig_ft2),
        ],
    );
    (m, mixing)
}
