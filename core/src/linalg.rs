//! Dense linear algebra for tall-skinny problems: Householder QR, one-sided Jacobi SVD,
//! cyclic Jacobi eigh and QR least squares. Every reduction runs in a fixed order.

/// Column-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_row_major(rows: usize, cols: usize, v: &[f64]) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m[(i, j)] = v[i * cols + j];
            }
        }
        m
    }

    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Two distinct columns, both mutable.
    fn cols_mut(&mut self, p: usize, q: usize) -> (&mut [f64], &mut [f64]) {
        let r = self.rows;
        let (a, b) = self.data.split_at_mut(q * r);
        (&mut a[p * r..(p + 1) * r], &mut b[..r])
    }

    pub fn t(&self) -> Mat {
        let mut m = Mat::zeros(self.cols, self.rows);
        for j in 0..self.cols {
            for i in 0..self.rows {
                m[(j, i)] = self[(i, j)];
            }
        }
        m
    }
}

impl std::ops::Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i + j * self.rows]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i + j * self.rows]
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Rotate two columns: (p, q) <- (c p - s q, s p + c q).
fn rotate(p: &mut [f64], q: &mut [f64], c: f64, s: f64) {
    for (x, y) in p.iter_mut().zip(q.iter_mut()) {
        let (a, b) = (*x, *y);
        *x = c * a - s * b;
        *y = s * a + c * b;
    }
}

/// Householder QR of an m x n matrix, m >= n. Reflector j lives in rows j.. of column j
/// (v_j = 1 not implied: stored in full), R's diagonal in `diag`, R's upper part above it.
pub struct Qr {
    store: Mat,
    beta: Vec<f64>,
    diag: Vec<f64>,
}

/// Apply `f` to each `m`-long column, in parallel for large matrices. Columns are independent,
/// so results do not depend on the thread count.
fn each_col(data: &mut [f64], m: usize, f: impl Fn(&mut [f64]) + Sync) {
    #[cfg(feature = "parallel")]
    if data.len() >= 1 << 16 {
        use rayon::prelude::*;
        return data.par_chunks_mut(m).for_each(&f);
    }
    data.chunks_mut(m).for_each(f);
}

impl Qr {
    pub fn new(a: &Mat) -> Self {
        let (m, n) = (a.rows, a.cols);
        assert!(m >= n, "Qr needs rows >= cols");
        let mut s = a.clone();
        let mut beta = vec![0.0; n];
        let mut diag = vec![0.0; n];
        for j in 0..n {
            let x = &mut s.col_mut(j)[j..];
            let norm = dot(x, x).sqrt();
            if norm == 0.0 {
                continue;
            }
            let alpha = if x[0] > 0.0 { -norm } else { norm };
            x[0] -= alpha;
            beta[j] = 1.0 / (-alpha * x[0]); // 2 / |v|^2, since |v|^2 = 2 (norm^2 - alpha x0)
            diag[j] = alpha;
            let (head, tail) = s.data.split_at_mut((j + 1) * m);
            let v = &head[j * m + j..];
            let bj = beta[j];
            each_col(tail, m, |col| {
                let y = &mut col[j..];
                let f = bj * dot(v, y);
                for (yi, vi) in y.iter_mut().zip(v) {
                    *yi -= f * vi;
                }
            });
        }
        Self {
            store: s,
            beta,
            diag,
        }
    }

    fn reflect(&self, j: usize, y: &mut [f64]) {
        let v = &self.store.col(j)[j..];
        let y = &mut y[j..];
        let f = self.beta[j] * dot(v, y);
        for (yi, vi) in y.iter_mut().zip(v) {
            *yi -= f * vi;
        }
    }

    /// Q x for each column of an m-row matrix.
    fn apply_q(&self, x: &mut Mat) {
        each_col(&mut x.data, x.rows, |col| {
            for j in (0..self.beta.len()).rev() {
                self.reflect(j, col);
            }
        });
    }

    /// Least-squares solution of `a b = y` for one right-hand side (full-rank a); y is overwritten.
    pub fn solve(&self, y: &mut [f64]) -> Vec<f64> {
        for j in 0..self.beta.len() {
            self.reflect(j, y);
        }
        let n = self.diag.len();
        let mut b = vec![0.0; n];
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|k| self.store[(i, k)] * b[k]).sum();
            b[i] = (y[i] - s) / self.diag[i];
        }
        b
    }

    pub fn r(&self) -> Mat {
        let n = self.diag.len();
        let mut r = Mat::zeros(n, n);
        for j in 0..n {
            r[(j, j)] = self.diag[j];
            for i in 0..j {
                r[(i, j)] = self.store[(i, j)];
            }
        }
        r
    }
}

/// Thin SVD `a = u diag(s) vt`, s descending. Signs are raw; callers apply numpy/sklearn rules.
pub fn svd(a: &Mat) -> (Mat, Vec<f64>, Mat) {
    if a.rows < a.cols {
        let (u, s, vt) = svd(&a.t());
        return (vt.t(), s, u.t());
    }
    let (u, s, vt) = svd_tall(a, true);
    (u.unwrap(), s, vt)
}

/// Singular values and right singular vectors only, for rows >= cols (skips forming U).
pub fn svd_values(a: &Mat) -> (Vec<f64>, Mat) {
    let (_, s, vt) = svd_tall(a, false);
    (s, vt)
}

fn svd_tall(a: &Mat, want_u: bool) -> (Option<Mat>, Vec<f64>, Mat) {
    let (m, n) = (a.rows, a.cols);
    assert!(m >= n, "svd_tall needs rows >= cols");
    let qr = Qr::new(a);
    let mut w = qr.r();
    let mut v = Mat::identity(n);
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (alpha, beta, gamma) = (
                    dot(w.col(p), w.col(p)),
                    dot(w.col(q), w.col(q)),
                    dot(w.col(p), w.col(q)),
                );
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta.abs() > 1e150 {
                    0.5 / zeta
                } else {
                    let t = 1.0 / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    if zeta < 0.0 { -t } else { t }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let (wp, wq) = w.cols_mut(p, q);
                rotate(wp, wq, c, c * t);
                let (vp, vq) = v.cols_mut(p, q);
                rotate(vp, vq, c, c * t);
            }
        }
        if !rotated {
            break;
        }
    }
    let norms: Vec<f64> = (0..n).map(|j| dot(w.col(j), w.col(j)).sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let mut vt = Mat::zeros(n, n);
    for (k, &j) in order.iter().enumerate() {
        for i in 0..n {
            vt[(k, i)] = v[(i, j)];
        }
    }
    let u = want_u.then(|| {
        let mut u = Mat::zeros(m, n);
        for (k, &j) in order.iter().enumerate() {
            // ponytail: a zero singular value leaves a zero U column; complete the basis if
            // rank-deficient data appears
            if norms[j] > 0.0 {
                for i in 0..n {
                    u[(i, k)] = w[(i, j)] / norms[j];
                }
            }
        }
        qr.apply_q(&mut u);
        u
    });
    (u, order.iter().map(|&j| norms[j]).collect(), vt)
}

/// `a b`.
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows);
    let mut c = Mat::zeros(a.rows, b.cols);
    for j in 0..b.cols {
        let cj = &mut c.data[j * a.rows..(j + 1) * a.rows];
        for k in 0..a.cols {
            let bkj = b[(k, j)];
            for (ci, ai) in cj.iter_mut().zip(a.col(k)) {
                *ci += ai * bkj;
            }
        }
    }
    c
}

/// Symmetric eigendecomposition, eigenvalues ascending (`scipy.linalg.eigh` order), vectors in columns.
pub fn eigh(a: &Mat) -> (Vec<f64>, Mat) {
    let n = a.rows;
    let mut a = a.clone();
    let mut v = Mat::identity(n);
    let frob2 = dot(&a.data, &a.data);
    for _sweep in 0..100 {
        let off2: f64 = (0..n)
            .flat_map(|j| (0..n).filter(move |&i| i != j).map(move |i| (i, j)))
            .map(|(i, j)| a[(i, j)] * a[(i, j)])
            .sum();
        if off2 <= f64::EPSILON * f64::EPSILON * frob2 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[(p, q)];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    let t = 1.0 / (theta.abs() + (theta * theta + 1.0).sqrt());
                    if theta < 0.0 { -t } else { t }
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let (cp, cq) = a.cols_mut(p, q);
                rotate(cp, cq, c, s);
                for k in 0..n {
                    let (x, y) = (a[(p, k)], a[(q, k)]);
                    a[(p, k)] = c * x - s * y;
                    a[(q, k)] = s * x + c * y;
                }
                let (vp, vq) = v.cols_mut(p, q);
                rotate(vp, vq, c, s);
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[(i, i)].total_cmp(&a[(j, j)]));
    let mut vs = Mat::zeros(n, n);
    for (k, &j) in order.iter().enumerate() {
        vs.col_mut(k).copy_from_slice(v.col(j));
    }
    (order.iter().map(|&j| a[(j, j)]).collect(), vs)
}

/// Least squares `x b = y` (rows >= cols): QR when x has full rank, else numpy's minimum-norm
/// SVD solution (singular values below rcond = eps * max(rows, cols) times the largest dropped).
pub fn lstsq(x: &Mat, y: &Mat) -> Mat {
    let qr = Qr::new(x);
    let rcond = f64::EPSILON * x.rows.max(x.cols) as f64;
    let dmax = qr.diag.iter().fold(0.0f64, |m, d| m.max(d.abs()));
    if qr.diag.iter().any(|d| d.abs() <= rcond * dmax) {
        let (u, s, vt) = svd(x);
        let mut uty = matmul(&u.t(), y);
        for (i, &si) in s.iter().enumerate() {
            let inv = if si > rcond * s[0] { 1.0 / si } else { 0.0 };
            (0..uty.cols).for_each(|c| uty[(i, c)] *= inv);
        }
        return matmul(&vt.t(), &uty);
    }
    let mut b = Mat::zeros(x.cols, y.cols);
    for c in 0..y.cols {
        let mut col = y.col(c).to_vec();
        b.col_mut(c).copy_from_slice(&qr.solve(&mut col));
    }
    b
}

/// `y - x lstsq(x, y)`.
pub fn residual(x: &Mat, y: &Mat) -> Mat {
    let fitted = matmul(x, &lstsq(x, y));
    let mut r = y.clone();
    r.data
        .iter_mut()
        .zip(&fitted.data)
        .for_each(|(a, b)| *a -= b);
    r
}
