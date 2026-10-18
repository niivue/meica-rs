//! sklearn 1.8 `FastICA(algorithm="parallel", fun="logcosh", whiten="unit-variance",
//! whiten_solver="svd")` and tedana's `f_ica` restart loop.
//!
//! Sums follow numpy: pairwise along a memory-contiguous axis, sequential otherwise.

use crate::linalg::{Mat, eigh, matmul, svd, svd_values};
use crate::math::tanh;
use crate::rng::RandomState;
use crate::stats::{mean_std, sum, zscore_columns};

pub struct Fit {
    /// Unmixing in the original space, `W K` (components x features).
    pub components: Mat,
    /// Whitening `K` (components x features).
    pub whitening: Mat,
    pub n_iter: usize,
    pub converged: bool,
}

/// `(u / sqrt(s)) (u^T w)` from `eigh(w w^T)`: numpy's multi_dot takes A(BC) for equal costs.
fn sym_decorrelation(w: &Mat) -> Mat {
    let (s, u) = eigh(&matmul(w, &w.t()));
    let mut scaled = u.clone();
    for (k, &sk) in s.iter().enumerate() {
        let inv = 1.0 / sk.max(f64::MIN_POSITIVE).sqrt();
        scaled.col_mut(k).iter_mut().for_each(|v| *v *= inv);
    }
    matmul(&scaled, &matmul(&u.t(), w))
}

/// `FastICA(...).fit(x)` with x samples x features (voxels x time).
pub fn fastica(x: &Mat, n_comp: usize, seed: u32, max_iter: usize, tol: f64) -> Fit {
    let (n_samples, n_features) = (x.rows, x.cols);
    // XT -= XT.mean(axis=-1): XT is a transposed view, so the voxel sums are sequential.
    let mut xc = x.clone();
    for t in 0..n_features {
        let mean = x.col(t).iter().fold(-0.0, |a, b| a + b) / n_samples as f64;
        xc.col_mut(t).iter_mut().for_each(|v| *v -= mean);
    }
    // u, d = svd(XT): XT's left singular vectors are xc's right ones; u *= sign(u[0]).
    let (d, vt) = svd_values(&xc);
    let mut k = Mat::zeros(n_comp, n_features);
    for c in 0..n_comp {
        let sign = vt[(c, 0)].signum();
        for t in 0..n_features {
            k[(c, t)] = vt[(c, t)] * sign / d[c];
        }
    }
    // X1 = K XT * sqrt(n_samples), held transposed (voxels x components).
    let mut x1 = matmul(&xc, &k.t());
    let root = (n_samples as f64).sqrt();
    x1.data.iter_mut().for_each(|v| *v *= root);

    let mut rs = RandomState::new(seed);
    let mut w_init = Mat::zeros(n_comp, n_comp);
    for i in 0..n_comp {
        for j in 0..n_comp {
            w_init[(i, j)] = rs.normal();
        }
    }
    let mut w = sym_decorrelation(&w_init);
    let p = n_samples as f64;
    let (mut n_iter, mut converged) = (0, false);
    for ii in 0..max_iter {
        let mut gx = matmul(&x1, &w.t());
        gx.data.iter_mut().for_each(|v| *v = tanh(*v));
        let g_wtx: Vec<f64> = (0..n_comp)
            .map(|c| {
                sum(&gx
                    .col(c)
                    .iter()
                    .map(|g| 1.0 * (1.0 - g * g))
                    .collect::<Vec<_>>())
                    / p
            })
            .collect();
        let mut inner = matmul(&gx.t(), &x1);
        for i in 0..n_comp {
            for j in 0..n_comp {
                inner[(i, j)] = inner[(i, j)] / p - g_wtx[i] * w[(i, j)];
            }
        }
        let w1 = sym_decorrelation(&inner);
        let lim = (0..n_comp)
            .map(|i| {
                ((0..n_comp)
                    .map(|j| w1[(i, j)] * w[(i, j)])
                    .sum::<f64>()
                    .abs()
                    - 1.0)
                    .abs()
            })
            .fold(f64::NEG_INFINITY, f64::max);
        w = w1;
        n_iter = ii + 1;
        if lim < tol {
            converged = true;
            break;
        }
    }
    // unit-variance: S = (W K) XT; each source's std (pairwise, its axis is contiguous) rescales W.
    let s = matmul(&xc, &matmul(&w, &k).t());
    for c in 0..n_comp {
        let (_, std) = mean_std(s.col(c));
        for j in 0..n_comp {
            w[(c, j)] /= std;
        }
    }
    Fit {
        components: matmul(&w, &k),
        whitening: k,
        n_iter,
        converged,
    }
}

/// `scipy.linalg.pinv` (SVD, rtol = max(M, N) eps).
pub fn pinv(a: &Mat) -> Mat {
    let (u, s, vt) = svd(a);
    let cutoff = s.iter().fold(0.0f64, |m, &v| m.max(v)) * a.rows.max(a.cols) as f64 * f64::EPSILON;
    let mut b = Mat::zeros(a.cols, a.rows);
    for (k, &sk) in s.iter().enumerate().filter(|(_, sk)| **sk > cutoff) {
        for i in 0..a.cols {
            for j in 0..a.rows {
                b[(i, j)] += vt[(k, i)] / sk * u[(j, k)];
            }
        }
    }
    b
}

pub struct Ica {
    /// z-scored mixing matrix (time x components), before tedana's metric sign flips and sort.
    pub mixing: Mat,
    /// Seed of the returned decomposition.
    pub seed: u32,
    pub fit: Fit,
}

/// tedana `f_ica`: refit with seed + 1 when FastICA does not converge, up to `max_restart`
/// times. As in tedana, the last two attempts reuse the same seed.
pub fn f_ica(x: &Mat, n_comp: usize, seed: u32, max_iter: usize, max_restart: usize) -> Ica {
    let mut seed = seed;
    let mut fit = fastica(x, n_comp, seed, max_iter, 1e-4);
    for attempt in 0..max_restart {
        if fit.converged {
            break;
        }
        if (attempt as i64) < max_restart as i64 - 1 {
            seed += 1;
        }
        fit = fastica(x, n_comp, seed, max_iter, 1e-4);
    }
    let mixing = zscore_columns(&pinv(&fit.components));
    Ica { mixing, seed, fit }
}
