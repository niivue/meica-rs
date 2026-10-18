//! Complex FFT with numpy's sign and scaling: radix-2 for powers of two, Bluestein otherwise.
//! Twiddles come from `sincos_2pi` with exact integer reduction, not a recurrence.

use crate::math::sincos_2pi;

/// Precomputed tables for one transform length: radix-2 twiddles, or Bluestein's chirp and its
/// transform on the padded power-of-two length.
struct Plan {
    n: usize,
    tw: Vec<(f64, f64)>,
    chirp: Vec<(f64, f64)>,
    chirp_fft: (Vec<f64>, Vec<f64>),
    inner: Option<Box<Plan>>,
}

impl Plan {
    fn new(n: usize) -> Self {
        let tw = if n.is_power_of_two() {
            (0..n / 2).map(|j| sincos_2pi(j as u64, n as u64)).collect()
        } else {
            Vec::new()
        };
        let mut plan = Plan {
            n,
            tw,
            chirp: Vec::new(),
            chirp_fft: (Vec::new(), Vec::new()),
            inner: None,
        };
        if n > 1 && !n.is_power_of_two() {
            let m = (2 * n - 1).next_power_of_two();
            let nn = 2 * n as u64;
            // chirp w_k = e^{-i pi k^2 / n}
            plan.chirp = (0..n as u64)
                .map(|k| {
                    let (s, c) = sincos_2pi((k * k) % nn, nn);
                    (c, -s)
                })
                .collect();
            let (mut br, mut bi) = (vec![0.0; m], vec![0.0; m]);
            for (k, &(wr, wi)) in plan.chirp.iter().enumerate() {
                br[k] = wr;
                bi[k] = -wi;
                if k > 0 {
                    br[m - k] = wr;
                    bi[m - k] = -wi;
                }
            }
            let inner = Plan::new(m);
            inner.radix2(&mut br, &mut bi);
            plan.chirp_fft = (br, bi);
            plan.inner = Some(Box::new(inner));
        }
        plan
    }

    fn radix2(&self, re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for j in 0..len / 2 {
                    let (s, c) = self.tw[j * step];
                    let (a, b) = (start + j, start + j + len / 2);
                    // b * e^{-i theta}
                    let (br, bi) = (re[b] * c + im[b] * s, im[b] * c - re[b] * s);
                    re[b] = re[a] - br;
                    im[b] = im[a] - bi;
                    re[a] += br;
                    im[a] += bi;
                }
            }
            len *= 2;
        }
    }

    /// Forward DFT in place: X_k = sum_n x_n e^{-2 pi i k n / N}.
    pub fn run(&self, re: &mut [f64], im: &mut [f64]) {
        debug_assert_eq!(re.len(), self.n);
        let n = self.n;
        if n <= 1 {
            return;
        }
        let Some(inner) = &self.inner else {
            return self.radix2(re, im);
        };
        let m = inner.n;
        let (ar, ai) = (&mut vec![0.0; m], &mut vec![0.0; m]);
        for k in 0..n {
            let (wr, wi) = self.chirp[k];
            ar[k] = re[k] * wr - im[k] * wi;
            ai[k] = re[k] * wi + im[k] * wr;
        }
        inner.radix2(ar, ai);
        let (br, bi) = &self.chirp_fft;
        for k in 0..m {
            let (x, y) = (ar[k] * br[k] - ai[k] * bi[k], ar[k] * bi[k] + ai[k] * br[k]);
            ar[k] = x;
            ai[k] = -y; // conjugate: inverse FFT via the forward one
        }
        inner.radix2(ar, ai);
        for k in 0..n {
            let (cr, ci) = (ar[k] / m as f64, -ai[k] / m as f64);
            let (wr, wi) = self.chirp[k];
            re[k] = cr * wr - ci * wi;
            im[k] = cr * wi + ci * wr;
        }
    }
}

/// Apply `f` to each `len`-long chunk of `re`/`im` (in parallel with the `parallel` feature).
fn each_line(
    re: &mut [f64],
    im: &mut [f64],
    len: usize,
    f: impl Fn(&mut [f64], &mut [f64]) + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        re.par_chunks_mut(len)
            .zip(im.par_chunks_mut(len))
            .for_each(|(r, i)| f(r, i));
    }
    #[cfg(not(feature = "parallel"))]
    re.chunks_mut(len)
        .zip(im.chunks_mut(len))
        .for_each(|(r, i)| f(r, i));
}

/// N-dimensional forward DFT of a C-order (last axis fastest) array, like `numpy.fft.fftn`.
/// Each axis's lines are gathered into a contiguous buffer and transformed independently, so
/// the result does not depend on the thread count.
pub fn fftn(re: &mut [f64], im: &mut [f64], shape: &[usize]) {
    for axis in 0..shape.len() {
        let len = shape[axis];
        let plan = Plan::new(len);
        let stride: usize = shape[axis + 1..].iter().product();
        if stride == 1 {
            each_line(re, im, len, |r, i| plan.run(r, i));
            continue;
        }
        let outer = re.len() / (len * stride);
        let (mut lr, mut li) = (vec![0.0; re.len()], vec![0.0; re.len()]);
        let base = |l: usize| (l / stride) * len * stride + l % stride;
        for l in 0..outer * stride {
            for k in 0..len {
                lr[l * len + k] = re[base(l) + k * stride];
                li[l * len + k] = im[base(l) + k * stride];
            }
        }
        each_line(&mut lr, &mut li, len, |r, i| plan.run(r, i));
        for l in 0..outer * stride {
            for k in 0..len {
                re[base(l) + k * stride] = lr[l * len + k];
                im[base(l) + k * stride] = li[l * len + k];
            }
        }
    }
}

/// N-dimensional inverse DFT scaled by 1/N (`numpy.fft.ifftn`).
pub fn ifftn(re: &mut [f64], im: &mut [f64], shape: &[usize]) {
    let n = re.len() as f64;
    im.iter_mut().for_each(|v| *v = -*v);
    fftn(re, im, shape);
    re.iter_mut().for_each(|v| *v /= n);
    im.iter_mut().for_each(|v| *v = -*v / n);
}
