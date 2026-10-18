//! Brain masks: nilearn's `compute_epi_mask` and tedana's dropout adaptive mask.
//!
//! 3D volumes are in NIfTI order (x fastest). Masked voxel lists follow numpy's C order over
//! (x, y, z), z fastest, as tedana's boolean indexing produces them.

use crate::math::{exp, ln};
use crate::stats::{percentile_higher, sum, sum_f32};

pub type Dims = [usize; 3];

/// NIfTI indices of the voxels in `mask`, in numpy C order.
pub fn c_order(d: Dims, mask: &[bool]) -> Vec<usize> {
    let mut out = Vec::new();
    for x in 0..d[0] {
        for y in 0..d[1] {
            for z in 0..d[2] {
                let i = x + d[0] * (y + d[1] * z);
                if mask[i] {
                    out.push(i);
                }
            }
        }
    }
    out
}

/// The 6-connected neighbours of voxel i that lie inside the grid.
pub(crate) fn neighbours(d: Dims, i: usize) -> impl Iterator<Item = usize> {
    let (x, y, z) = (i % d[0], (i / d[0]) % d[1], i / (d[0] * d[1]));
    let (sx, sy) = (1, d[0]);
    let sz = d[0] * d[1];
    [
        (x > 0).then(|| i - sx),
        (x + 1 < d[0]).then(|| i + sx),
        (y > 0).then(|| i - sy),
        (y + 1 < d[1]).then(|| i + sy),
        (z > 0).then(|| i - sz),
        (z + 1 < d[2]).then(|| i + sz),
    ]
    .into_iter()
    .flatten()
}

/// `scipy.ndimage.binary_erosion` with the 6-connected cross and border_value 0.
fn erode(d: Dims, m: &[bool], iterations: usize) -> Vec<bool> {
    let mut m = m.to_vec();
    for _ in 0..iterations {
        m = (0..m.len())
            .map(|i| m[i] && neighbours(d, i).count() == 6 && neighbours(d, i).all(|j| m[j]))
            .collect();
    }
    m
}

fn dilate(d: Dims, m: &[bool], iterations: usize) -> Vec<bool> {
    let mut m = m.to_vec();
    for _ in 0..iterations {
        m = (0..m.len())
            .map(|i| m[i] || neighbours(d, i).any(|j| m[j]))
            .collect();
    }
    m
}

/// nilearn `largest_connected_component` (6-connected). Labels follow scipy's C-order scan,
/// so a size tie keeps the component whose first voxel comes first.
fn largest_component(d: Dims, m: &[bool]) -> Vec<bool> {
    let mut label = vec![0usize; m.len()];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for seed in c_order(d, m) {
        if label[seed] != 0 {
            continue;
        }
        let l = sizes.len();
        sizes.push(0);
        label[seed] = l;
        stack.push(seed);
        while let Some(i) = stack.pop() {
            sizes[l] += 1;
            for j in neighbours(d, i) {
                if m[j] && label[j] == 0 {
                    label[j] = l;
                    stack.push(j);
                }
            }
        }
    }
    let best = (1..sizes.len()).fold(0, |b, l| if b == 0 || sizes[l] > sizes[b] { l } else { b });
    label.iter().map(|&l| l != 0 && l == best).collect()
}

/// nilearn `smooth_array(fwhm=1)` with an identity affine: scipy `gaussian_filter1d` along
/// x, y, z in turn, reflect mode, accumulating as scipy's symmetric-kernel loop does (outer taps
/// first, fused).
pub fn smooth_fwhm1(v: &mut [f64], d: Dims) {
    let sigma = 1.0 / (8.0 * ln(2.0)).sqrt();
    let radius = (4.0 * sigma + 0.5) as usize;
    let phi: Vec<f64> = (-(radius as i64)..=radius as i64)
        .map(|x| exp(-0.5 / (sigma * sigma) * (x * x) as f64))
        .collect();
    let total = sum(&phi);
    let w: Vec<f64> = phi[radius..].iter().map(|p| p / total).collect(); // w[j] = weight at offset +-j
    let strides = [1, d[0], d[0] * d[1]];
    for axis in 0..3 {
        let (n, stride) = (d[axis] as i64, strides[axis]);
        let mut line = vec![0.0; d[axis]];
        for start in 0..v.len() {
            if (start / stride) % d[axis] != 0 {
                continue; // not the first voxel of a line along this axis
            }
            for (k, l) in line.iter_mut().enumerate() {
                *l = v[start + k * stride];
            }
            let at = |mut i: i64| {
                while i < 0 || i >= n {
                    i = if i < 0 { -i - 1 } else { 2 * n - i - 1 };
                }
                line[i as usize]
            };
            for k in 0..n {
                let mut acc = line[k as usize] * w[0];
                for j in (1..=radius as i64).rev() {
                    // scipy's arm64 build contracts this multiply-add into an FMA.
                    acc = (at(k - j) + at(k + j)).mul_add(w[j as usize], acc);
                }
                v[start + k as usize * stride] = acc;
            }
        }
    }
}

/// nilearn `compute_epi_mask` with its defaults, for a 4D series in NIfTI order.
pub fn epi_mask(data: impl Fn(usize) -> f64, d: Dims, n_vol: usize) -> Vec<bool> {
    let nvox = d[0] * d[1] * d[2];
    let mut ts = vec![0.0; n_vol];
    let mut mean: Vec<f64> = (0..nvox)
        .map(|i| {
            for (t, v) in ts.iter_mut().enumerate() {
                *v = data(i + nvox * t);
            }
            let m = sum(&ts) / n_vol as f64;
            if m.is_finite() { m } else { 0.0 }
        })
        .collect();
    smooth_fwhm1(&mut mean, d);
    let mut sorted = mean.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let lower = (0.2 * n as f64).floor() as usize;
    let upper = ((0.85 * n as f64).floor() as usize).min(n - 1);
    let mut ia = 0;
    for k in 0..upper - lower {
        if sorted[lower + k + 1] - sorted[lower + k] > sorted[lower + ia + 1] - sorted[lower + ia] {
            ia = k;
        }
    }
    let threshold = 0.5 * (sorted[ia + lower] + sorted[ia + lower + 1]);
    let mask: Vec<bool> = mean.iter().map(|&m| m >= threshold).collect();
    let mask = erode(d, &mask, 2);
    let mask = if mask.iter().any(|&b| b) {
        largest_component(d, &mask)
    } else {
        mask
    };
    erode(d, &dilate(d, &mask, 4), 2)
}

/// tedana `make_adaptive_mask(methods=["dropout"], threshold=1)` on float32 data laid out
/// (numpy's float32 mean divides by an intp count, so through a float64 loop)
/// voxel x echo x time. Returns the number of good echoes per voxel.
pub fn adaptive_mask(data: &[f32], n_echo: usize, n_vol: usize) -> Vec<i32> {
    let n_vox = data.len() / (n_echo * n_vol);
    let series = |v: usize, e: usize| &data[(v * n_echo + e) * n_vol..(v * n_echo + e + 1) * n_vol];
    let means: Vec<f32> = (0..n_vox * n_echo)
        .map(|k| (sum_f32(series(k / n_echo, k % n_echo)) as f64 / n_vol as f64) as f32)
        .collect();
    let first: Vec<f64> = (0..n_vox)
        .map(|v| means[v * n_echo] as f64)
        .filter(|&m| m != 0.0)
        .collect();
    if first.is_empty() {
        return vec![0; n_vox];
    }
    let perc = percentile_higher(&first, 33.0) as f32;
    // Threshold each echo at a third of the percentile voxel's mean; on ties keep the voxel
    // with the largest summed thresholds.
    let mut lthrs: Vec<f32> = Vec::new();
    let mut best = f32::NEG_INFINITY;
    for v in (0..n_vox).filter(|&v| means[v * n_echo] == perc) {
        let t: Vec<f32> = means[v * n_echo..(v + 1) * n_echo]
            .iter()
            .map(|m| m / 3.0)
            .collect();
        let s = t.iter().fold(-0.0f32, |a, b| a + b);
        if s > best {
            (best, lthrs) = (s, t);
        }
    }
    (0..n_vox)
        .map(|v| {
            let mut base = 0;
            for e in 0..n_echo {
                if base == e as i32 && series(v, e).iter().all(|&x| x > 0.0) {
                    base = e as i32 + 1;
                }
            }
            let dropout = (0..n_echo)
                .filter(|&e| means[v * n_echo + e].abs() > lthrs[e])
                .map(|e| e as i32 + 1)
                .max()
                .unwrap_or(0);
            base.min(dropout)
        })
        .collect()
}
