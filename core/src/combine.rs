//! tedana `make_optcom` with `combmode="t2s"` (Posse et al. 1999).

use crate::decay::fit_echoes;
use crate::math::exp;

/// Optimally combined series (voxel x time), weights te * exp(-te / T2*) over each voxel's
/// fitted echoes. Voxels with masksum 0 stay 0.
pub fn optcom(data: &[f32], tes: &[f64], masksum: &[i32], t2s: &[f64], n_vol: usize) -> Vec<f64> {
    let n_echo = tes.len();
    let mut out = vec![0.0; masksum.len() * n_vol];
    for (v, &m) in masksum.iter().enumerate() {
        if m <= 0 {
            continue;
        }
        let k = fit_echoes(m);
        let w: Vec<f64> = tes[..k].iter().map(|te| te * exp(-te / t2s[v])).collect();
        let total = w.iter().fold(-0.0, |a, b| a + b);
        for t in 0..n_vol {
            let num = (0..k).fold(-0.0, |a, e| {
                a + data[(v * n_echo + e) * n_vol + t] as f64 * w[e]
            });
            out[v * n_vol + t] = num / total;
        }
    }
    out
}
