//! The scipy.stats distribution functions tedana's default path calls.

use crate::linalg::Mat;
use crate::math::{exp, lbeta, ln, ln1p};

macro_rules! pairwise_sum {
    ($name:ident, $t:ty) => {
        /// numpy's pairwise summation (8 accumulators, 128-element blocks): `np.sum` over a
        /// contiguous axis, bit for bit.
        pub fn $name(a: &[$t]) -> $t {
            let n = a.len();
            if n < 8 {
                return a.iter().fold(-0.0, |r, &v| r + v);
            }
            if n <= 128 {
                let mut r: [$t; 8] = a[..8].try_into().unwrap();
                let mut i = 8;
                while i < n - n % 8 {
                    for k in 0..8 {
                        r[k] += a[i + k];
                    }
                    i += 8;
                }
                let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for &v in &a[i..] {
                    res += v;
                }
                return res;
            }
            let half = n / 2 - (n / 2) % 8;
            $name(&a[..half]) + $name(&a[half..])
        }
    };
}
pairwise_sum!(sum, f64);
pairwise_sum!(sum_f32, f32);

/// `np.mean` and `np.std` of a contiguous series (pairwise sums).
pub fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = sum(v) / n;
    let sq: Vec<f64> = v.iter().map(|x| (x - mean) * (x - mean)).collect();
    (mean, (sum(&sq) / n).sqrt())
}

/// scipy `zscore` of a contiguous series.
pub fn zscore(y: &mut [f64]) {
    let (mean, std) = mean_std(y);
    y.iter_mut().for_each(|x| *x = (*x - mean) / std);
}

/// Subtract the (pairwise) mean.
pub fn demean(y: &mut [f64]) {
    let mean = sum(y) / y.len() as f64;
    y.iter_mut().for_each(|x| *x -= mean);
}

/// scipy `zscore(axis=0)` of a C-ordered matrix: sequential sums down each column.
pub fn zscore_columns(m: &Mat) -> Mat {
    let mut z = m.clone();
    let n = m.rows as f64;
    for c in 0..m.cols {
        let col = z.col_mut(c);
        let mean = col.iter().fold(-0.0, |a, b| a + b) / n;
        let std = (col.iter().fold(-0.0, |a, v| a + (v - mean) * (v - mean)) / n).sqrt();
        col.iter_mut().for_each(|v| *v = (*v - mean) / std);
    }
    z
}

/// `np.median` (NaN when empty).
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let h = s.len() / 2;
    if s.len() % 2 == 1 {
        s[h]
    } else {
        (-0.0 + s[h - 1] + s[h]) / 2.0
    }
}

/// Continued fraction for I_x(a, b) (modified Lentz), convergent for x < (a+1)/(a+b+2).
// ponytail: with a >> b near that bound it loses ~1e-12 (t CDF at df ~ 1e5, which only feeds the
// reported signal-noise_p); switch to Didonato-Morris BGRAT if a gated value ever needs it.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    d = 1.0 / if d.abs() < TINY { TINY } else { d };
    let mut h = d;
    for m in 1..10_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        for aa in [
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ] {
            d = 1.0 + aa * d;
            d = 1.0 / if d.abs() < TINY { TINY } else { d };
            c = 1.0 + aa / c;
            if c.abs() < TINY {
                c = TINY;
            }
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 4.0 * f64::EPSILON {
            return h;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b); `xc` must be 1 - x, computed without cancellation.
fn betai(a: f64, b: f64, x: f64, xc: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if xc <= 0.0 {
        return 1.0;
    }
    let lnx = if x > 0.5 { ln1p(-xc) } else { ln(x) };
    let lnxc = if xc > 0.5 { ln1p(-x) } else { ln(xc) };
    let front = exp(a * lnx + b * lnxc - lbeta(a, b));
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, xc) / b
    }
}

/// `scipy.special.betainc(a, b, x)`.
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    betai(a, b, x, 1.0 - x)
}

/// Student t CDF, `scipy.stats.t.cdf(t, df)`.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    if t == 0.0 {
        return 0.5;
    }
    let t2 = t * t;
    // Lower tail = 0.5 * I_{df/(df+t^2)}(df/2, 1/2), with both x and 1 - x formed directly.
    let tail = 0.5 * betai(0.5 * df, 0.5, df / (df + t2), t2 / (df + t2));
    if t < 0.0 { tail } else { 1.0 - tail }
}

/// `scipy.stats.f.ppf(q, 1, dfd)`, as t_{dfd} quantile squared, by bisection.
pub fn f_ppf_dfn1(q: f64, dfd: f64) -> f64 {
    let p = 0.5 * (1.0 + q);
    let mut hi = 1.0;
    while t_cdf(hi, dfd) < p {
        hi *= 2.0;
    }
    let mut lo = 0.0;
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            return hi * hi;
        }
        if t_cdf(mid, dfd) < p {
            lo = mid
        } else {
            hi = mid
        }
    }
}

/// `scipy.stats.rankdata(method="average")`: 1-based ranks, ties share their mean rank.
pub fn rankdata(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && x[order[j]] == x[order[i]] {
            j += 1;
        }
        let r = (i + 1 + j) as f64 / 2.0;
        for &k in &order[i..j] {
            ranks[k] = r;
        }
        i = j;
    }
    ranks
}

/// `np.percentile(x, q, method="higher")`: the element at the ceiling of numpy's virtual index.
pub fn percentile_higher(x: &[f64], q: f64) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let q = q / 100.0;
    let (alpha, beta) = (1.0, 1.0); // numpy's _compute_virtual_index for the discrete methods
    let virtual_index = s.len() as f64 * q + (alpha + q * (1.0 - alpha - beta)) - 1.0;
    s[(virtual_index.ceil().max(0.0) as usize).min(s.len() - 1)]
}

/// scipy `ttest_ind(a, b, equal_var=False)`: Welch's t and its two-sided p. NaN when either
/// sample has fewer than two values.
pub fn welch_ttest(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    if a.len() < 2 || b.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let moments = |x: &[f64], n: f64| {
        let m = sum(x) / n;
        let sq: Vec<f64> = x.iter().map(|v| (v - m) * (v - m)).collect();
        (m, sum(&sq) / (n - 1.0))
    };
    let ((m1, v1), (m2, v2)) = (moments(a, n1), moments(b, n2));
    let (vn1, vn2) = (v1 / n1, v2 / n2);
    let df = (vn1 + vn2) * (vn1 + vn2) / (vn1 * vn1 / (n1 - 1.0) + vn2 * vn2 / (n2 - 1.0));
    let t = (m1 - m2) / (vn1 + vn2).sqrt();
    (t, 2.0 * t_cdf(-t.abs(), df))
}
