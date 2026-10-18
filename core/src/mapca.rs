//! mapca 0.0.8 `MovingAveragePCA` (Li, Adali & Calhoun 2007, after GIFT), function by function.
//!
//! mapca reshapes volumes in Fortran order, so its sample rows are mask voxels in NIfTI order
//! (x fastest), not tedana's C order. Volumes it treats as 3D arrays are kept here in numpy's
//! C order (z fastest) so that their sums run in the same order.

use crate::fft::{fftn, ifftn};
use crate::linalg::{Mat, matmul, svd_values};
use crate::math::{exp, ln, pow};
use crate::stats::{mean_std, median, sum};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Criterion {
    Aic,
    Kic,
    Mdl,
}

impl Criterion {
    /// tedana's `--tedpca` name.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "aic" => Ok(Criterion::Aic),
            "kic" => Ok(Criterion::Kic),
            "mdl" => Ok(Criterion::Mdl),
            other => Err(format!("tedpca {other} is not ported (aic, kic or mdl)")),
        }
    }
}

pub struct MaPca {
    pub iid_subsample_input: Vec<f64>,
    /// np.round(mean(iid_subsample_input), 3).
    pub iid_mean: f64,
    pub subsample_depth: usize,
    pub effective_samples: f64,
    /// (number of components, criterion curve) for AIC, KIC and MDL.
    pub aic: (usize, Vec<f64>),
    pub kic: (usize, Vec<f64>),
    pub mdl: (usize, Vec<f64>),
    pub varex_90: usize,
    pub varex_95: usize,
    pub cumulative_varex: Vec<f64>,
    /// Kept components (rows, sklearn sign convention), their variances and variance ratios.
    pub components: Mat,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    /// Voxel weights `x components^T / variance` (rows as the input).
    pub u: Mat,
}

/// Sequential column sums, as numpy reduces axis 0 of a C-ordered 2D array.
fn column_mean(x: &Mat, j: usize) -> f64 {
    x.col(j).iter().fold(-0.0, |a, b| a + b) / x.rows as f64
}

/// sklearn `StandardScaler().fit_transform(x.T).T`: each row centred and scaled over its
/// columns, with numpy pairwise sums along the row and sklearn's two-pass variance.
fn standardize_rows(x: &mut Mat) {
    let n = x.cols as f64;
    let mut row = vec![0.0; x.cols];
    for i in 0..x.rows {
        for (j, r) in row.iter_mut().enumerate() {
            *r = x[(i, j)];
        }
        let total = sum(&row);
        let mean = total / n;
        let dev: Vec<f64> = row.iter().map(|v| v - mean).collect();
        let correction = sum(&dev);
        let sq: Vec<f64> = dev.iter().map(|d| d * d).collect();
        let var = (sum(&sq) - correction * correction / n) / n;
        // sklearn _is_constant_feature: such rows keep scale 1.
        let constant =
            var <= n * f64::EPSILON * var + (n * mean * f64::EPSILON) * (n * mean * f64::EPSILON);
        let scale = if constant { 1.0 } else { var.sqrt() };
        for j in 0..x.cols {
            x[(i, j)] = (x[(i, j)] - mean) / scale;
        }
    }
}

/// mapca `_icatb_svd`, returned in descending order: eigenvalues (s / sqrt(m - 1))^2 and
/// right singular vectors as columns.
fn icatb_svd(x: &Mat) -> (Vec<f64>, Mat) {
    let (s, vt) = svd_values(x);
    let scale = ((x.rows - 1) as f64).sqrt();
    let eig = s.iter().map(|v| (v / scale) * (v / scale)).collect();
    (eig, vt.t())
}

/// scipy `kurtosis(fisher=True, bias=True)` of each column, axis-0 sums sequential.
fn kurtosis(x: &Mat, j: usize) -> f64 {
    let mean = column_mean(x, j);
    let d: Vec<f64> = x.col(j).iter().map(|v| v - mean).collect();
    let n = x.rows as f64;
    let m2 = d.iter().fold(-0.0, |a, v| a + v * v) / n;
    let m4 = d.iter().fold(-0.0, |a, v| a + (v * v) * (v * v)) / n;
    if m2 <= (f64::EPSILON * mean) * (f64::EPSILON * mean) {
        return f64::NAN;
    }
    m4 / (m2 * m2) - 3.0
}

/// scipy.signal.windows.parzen(m), m odd.
fn parzen(m: usize) -> Vec<f64> {
    let mf = m as f64;
    let half = mf / 2.0;
    let cube = |v: f64| v * v * v; // ponytail: numpy calls libm pow(x, 3.0); this may differ by an ulp
    (0..m)
        .map(|i| {
            let n = -(mf - 1.0) / 2.0 + i as f64;
            let a = n.abs() / half;
            if n.abs() <= (mf - 1.0) / 4.0 {
                1.0 - 6.0 * (a * a) + 6.0 * cube(a)
            } else {
                2.0 * cube(1.0 - a)
            }
        })
        .collect()
}

/// `fftconvolve(v, flip(v))` in full mode for a C-order 3D array: |F|^2 on the zero-padded
/// 2d - 1 grid (just large enough that the circular correlation does not wrap), inverted, then
/// lag 0 moved to index d - 1 on each axis.
fn autocorrelation(v: &[f64], d: [usize; 3]) -> Vec<f64> {
    let p: [usize; 3] = std::array::from_fn(|a| 2 * d[a] - 1);
    let (mut re, mut im) = (vec![0.0; p[0] * p[1] * p[2]], vec![0.0; p[0] * p[1] * p[2]]);
    for x in 0..d[0] {
        for y in 0..d[1] {
            let (src, dst) = ((x * d[1] + y) * d[2], (x * p[1] + y) * p[2]);
            re[dst..dst + d[2]].copy_from_slice(&v[src..src + d[2]]);
        }
    }
    fftn(&mut re, &mut im, &p);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = *r * *r + *i * *i;
        *i = 0.0;
    }
    ifftn(&mut re, &mut im, &p);
    let lag = |a: usize, k: usize| (k + p[a] - (d[a] - 1)) % p[a];
    let mut out = Vec::with_capacity(re.len());
    for x in 0..p[0] {
        for y in 0..p[1] {
            for z in 0..p[2] {
                out.push(re[(lag(0, x) * p[1] + lag(1, y)) * p[2] + lag(2, z)]);
            }
        }
    }
    out
}

/// mapca `ent_rate_sp(data, sm_window=1)` for a C-order 3D array.
fn ent_rate_sp(v: &[f64], d: [usize; 3]) -> Result<f64, String> {
    if d.iter().any(|&n| n < 2) {
        return Err("Incorrect matrix dimensions.".into());
    }
    let (_, std) = mean_std(v);
    if std == 0.0 {
        return Err("Divide by zero encountered.".into());
    }
    let scaled: Vec<f64> = v.iter().map(|x| x / std).collect();
    let mut corr = autocorrelation(&scaled, d);
    let o: [usize; 3] = std::array::from_fn(|a| 2 * d[a] - 1);
    let tri = |a: usize, k: usize| (if k < d[a] { k + 1 } else { 2 * d[a] - 1 - k }) as i64;
    let windows: Vec<Vec<f64>> = (0..3)
        .map(|a| {
            let m = d[a].div_ceil(10);
            let mut w = vec![0.0; o[a]];
            w[d[a] - m - 1..d[a] + m].copy_from_slice(&parzen(2 * m + 1));
            w
        })
        .collect();
    for x in 0..o[0] {
        for y in 0..o[1] {
            for z in 0..o[2] {
                let k = (x * o[1] + y) * o[2] + z;
                corr[k] /= ((tri(0, x) * tri(1, y)) * tri(2, z)) as f64;
                // mapca fills the 3D window for slices 1..=2*d2-3 only.
                let w3 = if z == 0 || z == o[2] - 1 {
                    0.0
                } else {
                    windows[2][z]
                };
                corr[k] *= (windows[0][x] * windows[1][y]) * w3;
            }
        }
    }
    let mut im = vec![0.0; corr.len()];
    fftn(&mut corr, &mut im, &o);
    // abs, then fftshift: sum in the shifted C order.
    let mut mags = Vec::with_capacity(corr.len());
    for x in 0..o[0] {
        for y in 0..o[1] {
            for z in 0..o[2] {
                let s = |a: usize, k: usize| (k + o[a] - o[a] / 2) % o[a];
                let k = (s(0, x) * o[1] + s(1, y)) * o[2] + s(2, z);
                let m = (corr[k] * corr[k] + im[k] * im[k]).sqrt();
                mags.push(if m < 0.0001 { 0.0001 } else { m });
            }
        }
    }
    let logs: Vec<f64> = mags.iter().map(|&m| ln(m)).collect();
    Ok(0.5 * ln(2.0 * std::f64::consts::PI * exp(1.0)) + sum(&logs) / 2.0 / sum(&mags))
}

/// `data[::s, ::s, ::s]` of a C-order 3D array.
fn subsample(v: &[f64], d: [usize; 3], s: usize) -> (Vec<f64>, [usize; 3]) {
    let nd = d.map(|n| n.div_ceil(s));
    let mut out = Vec::with_capacity(nd[0] * nd[1] * nd[2]);
    for x in (0..d[0]).step_by(s) {
        for y in (0..d[1]).step_by(s) {
            for z in (0..d[2]).step_by(s) {
                out.push(v[(x * d[1] + y) * d[2] + z]);
            }
        }
    }
    (out, nd)
}

/// mapca `_est_indp_sp`: the subsampling depth (minus one) at which the entropy rate first
/// exceeds 1.41.
fn est_indp_sp(v: &[f64], d: [usize; 3]) -> Result<usize, String> {
    for j in 0..d.iter().min().unwrap() - 1 {
        let (sb, sd) = subsample(v, d, j + 1);
        if ent_rate_sp(&sb, sd)? > 1.41 {
            return Ok(j);
        }
    }
    Err("Ill conditioned data, can not estimate independent samples.".into())
}

/// mapca `_eigensp_adj`.
fn eigensp_adj(lam: &[f64], n: f64, p: usize) -> Vec<f64> {
    let pf = p as f64;
    let r = pf / n;
    let bp = (1.0 + r.sqrt()) * (1.0 + r.sqrt());
    let bm = (1.0 - r.sqrt()) * (1.0 - r.sqrt());
    let step = (bp - bm) / (5 * p - 1) as f64;
    // np.arange(bm, bp + step, step): length from ceil; numpy stores start and start + step, then
    // fills start + i * (second - first).
    let len = ((bp + step - bm) / step).ceil() as usize;
    let delta = (bm + step) - bm;
    let vv: Vec<f64> = (0..len)
        .map(|i| {
            if i == 1 {
                bm + step
            } else {
                bm + i as f64 * delta
            }
        })
        .collect();
    let gv: Vec<f64> = vv
        .iter()
        .map(|&v| 1.0 / (2.0 * std::f64::consts::PI * r * v) * ((v - bm) * (bp - v)).abs().sqrt())
        .collect();
    let mut gvd = vec![0.0; gv.len()];
    for i in 0..gv.len() {
        gvd[i] = gv[..i].iter().fold(0.0, |a, b| a + b); // Python's builtin sum
    }
    let peak = gvd.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    gvd.iter_mut().for_each(|g| *g /= peak);
    let mut emp: Vec<f64> = (1..=p)
        .map(|i| {
            let target = i as f64 / pf;
            let best = (0..gvd.len()).fold(0, |b, k| {
                if (target - gvd[k]).abs() < (target - gvd[b]).abs() {
                    k
                } else {
                    b
                }
            });
            vv[best]
        })
        .collect();
    emp.reverse();
    lam.iter().zip(&emp).map(|(l, e)| l / e).collect()
}

/// First k (1-based) where the criterion curve increases, else its length.
fn first_rise(curve: &[f64]) -> usize {
    curve
        .windows(2)
        .position(|w| w[1] - w[0] > 0.0)
        .map_or(curve.len(), |i| i + 1)
}

/// `MovingAveragePCA(criterion, normalize=True).fit(x)`: x is samples x time, rows the mask
/// voxels in NIfTI order; `positions` gives each row's NIfTI index in a grid of `dims`.
pub fn mapca(
    x: &Mat,
    dims: [usize; 3],
    positions: &[usize],
    criterion: Criterion,
) -> Result<MaPca, String> {
    let (n_samples, n_t) = (x.rows, x.cols);
    let mut x = x.clone();
    standardize_rows(&mut x);
    let (eig, v) = icatb_svd(&x);
    let data_n = matmul(&x, &v);
    let eig_mean = sum(&eig) / eig.len() as f64;
    let gauss: Vec<usize> = (0..n_t)
        .filter(|&j| {
            let mut k = kurtosis(&data_n, j);
            if k < 0.0 {
                k = 0.0;
            }
            if eig[j] > eig_mean {
                k = 1000.0;
            }
            k < 0.3 && k > 0.0 && eig[j] > f64::EPSILON
        })
        .collect();
    let dfs = eig.iter().filter(|&&e| e > f64::EPSILON).count();
    let mut idx: Vec<usize> = if gauss.len() >= 12 {
        let middle = (gauss.len() as f64 / 2.0).round_ties_even() as usize;
        [
            &gauss[..4],
            &gauss[middle - 1..middle + 3],
            &gauss[gauss.len() - 4..],
        ]
        .concat()
    } else {
        let min_tp = 12.min(dfs);
        (dfs - min_tp..dfs).collect()
    };
    idx.sort_unstable();
    idx.dedup();
    if idx.is_empty() {
        return Err("maPCA found no components to estimate the sampling depth from".into());
    }

    // C-order volume of one column of data_n.
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    let to_c = |i: usize| ((i % nx) * ny + (i / nx) % ny) * nz + i / (nx * ny);
    let volume = |col: &[f64]| {
        let mut vol = vec![0.0; nx * ny * nz];
        for (r, &i) in positions.iter().enumerate() {
            vol[to_c(i)] = col[r];
        }
        vol
    };
    let mut iid = Vec::new();
    for (i, &c) in idx.iter().enumerate() {
        iid.push(est_indp_sp(&volume(data_n.col(c)), dims)? as f64 + 1.0);
        if i > 6 {
            let head = &iid[..i];
            let m = median(head).round_ties_even();
            if head.iter().filter(|&&s| s == m).count() > 6 {
                iid.truncate(i);
                break;
            }
        }
    }
    let mut depth = median(&iid).round_ties_even() as usize;
    let cap = pow(n_samples as f64 / n_t as f64, 1.0 / 3.0).floor() as usize;
    if cap < depth {
        depth = cap;
    }
    let n = (n_samples as f64 / depth.pow(3) as f64).round_ties_even();

    let mut eig = eig;
    if depth != 1 {
        let mut grid_mask = vec![false; nx * ny * nz];
        for &i in positions {
            grid_mask[to_c(i)] = true;
        }
        let mask_f: Vec<f64> = grid_mask.iter().map(|&b| b as u8 as f64).collect();
        let (ms, sd) = subsample(&mask_f, dims, depth);
        // Subsampled mask voxels in Fortran (NIfTI) order of the subsampled grid.
        let mut keep = Vec::new();
        for z in 0..sd[2] {
            for y in 0..sd[1] {
                for xx in 0..sd[0] {
                    let c = (xx * sd[1] + y) * sd[2] + z;
                    if ms[c] != 0.0 {
                        keep.push(c);
                    }
                }
            }
        }
        let mut dat = Mat::zeros(keep.len(), n_t);
        for t in 0..n_t {
            let (s, _) = subsample(&volume(x.col(t)), dims, depth);
            for (r, &c) in keep.iter().enumerate() {
                dat[(r, t)] = s[c];
            }
        }
        standardize_rows(&mut dat);
        eig = icatb_svd(&dat).0;
    }
    let mut eig = eigensp_adj(&eig, n, eig.len());
    let floor = eig
        .iter()
        .filter(|&&e| e >= f64::EPSILON)
        .fold(f64::INFINITY, |a, &b| a.min(b));
    eig.iter_mut()
        .filter(|e| **e <= f64::EPSILON)
        .for_each(|e| *e = floor);

    let p = n_t;
    let (mut aic, mut kic, mut mdl) = (Vec::new(), Vec::new(), Vec::new());
    for k in 1..p {
        let tail = &eig[k..];
        let prod = tail
            .iter()
            .fold(1.0, |a, &e| a * pow(e, 1.0 / (p - k) as f64));
        let lh = ln(prod / (sum(tail) / tail.len() as f64));
        let mlh = 0.5 * n * (p - k) as f64 * lh;
        let df = 1.0 + 0.5 * k as f64 * (2 * p - k + 1) as f64;
        aic.push(-2.0 * mlh + 2.0 * df);
        kic.push(-2.0 * mlh + 3.0 * df);
        mdl.push(-mlh + 0.5 * df * ln(n));
    }
    let (n_aic, n_kic, n_mdl) = (first_rise(&aic), first_rise(&kic), first_rise(&mdl));
    let n_comp = match criterion {
        Criterion::Aic => n_aic,
        Criterion::Kic => n_kic,
        Criterion::Mdl => n_mdl,
    };

    // sklearn PCA(svd_solver="full") on the normalised data.
    let mut centred = x.clone();
    for j in 0..n_t {
        let m = column_mean(&x, j);
        centred.col_mut(j).iter_mut().for_each(|v| *v -= m);
    }
    let (s, mut vt) = svd_values(&centred);
    for k in 0..vt.rows {
        // svd_flip(u_based_decision=False): the largest |entry| of each Vt row is positive.
        let best = (0..vt.cols).fold(0, |b, j| {
            if vt[(k, j)].abs() > vt[(k, b)].abs() {
                j
            } else {
                b
            }
        });
        if vt[(k, best)] < 0.0 {
            for j in 0..vt.cols {
                vt[(k, j)] = -vt[(k, j)];
            }
        }
    }
    let ev: Vec<f64> = s.iter().map(|v| v * v / (n_samples - 1) as f64).collect();
    let total = sum(&ev);
    let ratio: Vec<f64> = ev.iter().map(|e| e / total).collect();
    let cumulative: Vec<f64> = ratio
        .iter()
        .scan(0.0, |a, &r| {
            *a += r;
            Some(*a)
        })
        .collect();
    let reach = |q: f64| cumulative.iter().position(|&c| c >= q).unwrap() + 1;
    let mut components = Mat::zeros(n_comp, n_t);
    for k in 0..n_comp {
        for j in 0..n_t {
            components[(k, j)] = vt[(k, j)];
        }
    }
    // PCA(copy=False) centred x in place, so mapca's u_ uses the centred data.
    let mut u = matmul(&centred, &components.t());
    for (k, e) in ev[..n_comp].iter().enumerate() {
        let inv = 1.0 / e;
        u.col_mut(k).iter_mut().for_each(|v| *v *= inv);
    }
    let iid_mean = (sum(&iid) / iid.len() as f64 * 1000.0).round_ties_even() / 1000.0;
    Ok(MaPca {
        iid_subsample_input: iid,
        iid_mean,
        subsample_depth: depth,
        effective_samples: n,
        aic: (n_aic, aic),
        kic: (n_kic, kic),
        mdl: (n_mdl, mdl),
        varex_90: reach(0.9),
        varex_95: reach(0.95),
        cumulative_varex: cumulative,
        components,
        explained_variance: ev[..n_comp].to_vec(),
        explained_variance_ratio: ratio[..n_comp].to_vec(),
        u,
    })
}
