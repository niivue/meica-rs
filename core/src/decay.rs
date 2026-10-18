//! tedana's log-linear monoexponential fit (`fit_loglinear`) and `modify_t2s_s0_maps`.

use crate::linalg::{Mat, Qr};
use crate::math::{exp, ln, ln_f32};

/// Echo count each voxel is fitted with: voxels with one good echo use the first two.
pub fn fit_echoes(masksum: i32) -> usize {
    masksum.max(2) as usize
}

/// T2* and S0 per voxel (voxel x echo x time f32 data, TEs in seconds). Voxels with masksum 0
/// get 0. Data enter the fit as float32 `log(|x| + 1)`, as numpy computes them.
pub fn fit_loglinear(
    data: &[f32],
    tes: &[f64],
    masksum: &[i32],
    n_vol: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_echo = tes.len();
    let qrs: Vec<Option<Qr>> = (0..=n_echo)
        .map(|k| {
            (k >= 2).then(|| {
                let mut x = Mat::zeros(k * n_vol, 2);
                for e in 0..k {
                    for t in 0..n_vol {
                        x[(e * n_vol + t, 0)] = 1.0;
                        x[(e * n_vol + t, 1)] = -tes[e];
                    }
                }
                Qr::new(&x)
            })
        })
        .collect();
    let (mut t2s, mut s0) = (vec![0.0; masksum.len()], vec![0.0; masksum.len()]);
    let mut y = Vec::with_capacity(n_echo * n_vol);
    for (v, &m) in masksum.iter().enumerate() {
        if m <= 0 {
            continue;
        }
        let k = fit_echoes(m);
        y.clear();
        y.extend(
            data[v * n_echo * n_vol..(v * n_echo + k) * n_vol]
                .iter()
                .map(|&d| ln_f32(d.abs() + 1.0) as f64),
        );
        let b = qrs[k].as_ref().unwrap().solve(&mut y);
        t2s[v] = 1.0 / b[1];
        s0[v] = exp(b[0]);
    }
    (t2s, s0)
}

/// Floors and NaN handling of tedana `modify_t2s_s0_maps` on the fitted (masksum > 0) voxels.
/// The limited maps it also returns are only written in verbose mode.
pub fn modify_maps(t2s: &mut [f64], s0: &mut [f64], tes: &[f64], masksum: &[i32]) {
    let floor = tes.iter().map(|t| -t).fold(f64::INFINITY, f64::min) / ln(f64::EPSILON);
    for (t, _) in t2s.iter_mut().zip(masksum).filter(|(_, m)| **m > 0) {
        if t.is_infinite() {
            *t = 0.5;
        }
        if *t <= 0.0 {
            *t = 0.001;
        }
        if tes.iter().any(|te| exp(-te / *t) == 0.0) {
            *t = floor;
        }
    }
    for (s, _) in s0.iter_mut().zip(masksum).filter(|(_, m)| **m > 0) {
        if s.is_nan() {
            *s = 0.0;
        }
    }
}
