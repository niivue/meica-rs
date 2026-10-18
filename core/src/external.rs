//! tedana's external-regressor metrics (metrics/external.py) and AROMA's high-frequency content
//! (metrics/frequency.py): F tests of the mixing on regressor models, max_rp_corr and HFC.

use crate::fft::fftn;
use crate::json::Json;
use crate::linalg::{Mat, residual};
use crate::rng::Pcg64;
use crate::stats::{betainc, zscore};

/// The external regressors TSV as pandas `read_table` loads it.
pub struct Regressors {
    pub names: Vec<String>,
    pub cols: Vec<Vec<f64>>,
}

impl Regressors {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let names: Vec<String> = lines
            .next()
            .ok_or("external regressors: empty file")?
            .split('\t')
            .map(|s| s.trim().to_string())
            .collect();
        // py_match recurses per character; bounded names keep it within wasm's ~1 MB stack.
        if let Some(n) = names.iter().find(|n| n.len() > 256) {
            return Err(format!(
                "external regressors: column name '{n:.40}…' is too long"
            ));
        }
        let mut cols = vec![Vec::new(); names.len()];
        for (row, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split('\t').collect();
            if cells.len() != names.len() {
                return Err(format!(
                    "external regressors: row {} has {} columns, the header {}",
                    row + 1,
                    cells.len(),
                    names.len()
                ));
            }
            for (c, cell) in cells.iter().enumerate() {
                let cell = cell.trim();
                // pandas' default NA strings.
                let v = match cell {
                    "" | "n/a" | "N/A" | "NA" | "NaN" | "nan" | "-nan" | "-NaN" | "#N/A"
                    | "null" | "NULL" | "None" | "<NA>" => f64::NAN,
                    _ => cell.parse().map_err(|_| {
                        format!(
                            "external regressors: '{cell}' in column {} is not a number",
                            names[c]
                        )
                    })?,
                };
                cols[c].push(v);
            }
        }
        Ok(Self { names, cols })
    }

    fn col(&self, name: &str) -> &[f64] {
        &self.cols[self.names.iter().position(|n| n == name).unwrap()]
    }
}

/// `validate_extern_regress` on the tree's `external_regressor_config` (no dummy scans): regex
/// names expanded to the sorted matching columns, partial models checked against their full
/// model, NaNs back- then forward-filled, and the row count matched to the volumes.
pub fn validate(reg: &mut Regressors, config: &mut Json, n_vol: usize) -> Result<(), String> {
    let Json::Arr(models) = config else {
        return Err("external_regressor_config must be a list".into());
    };
    let mut err = String::new();
    let expand = |names: &Json, err: &mut String| -> Result<Vec<String>, String> {
        let mut out = std::collections::BTreeSet::new();
        for t in names.as_arr().ok_or("regressor names must be a list")? {
            let t = t.as_str().ok_or("regressor names must be strings")?;
            if t.starts_with('^') {
                let mut found = false;
                for n in &reg.names {
                    if py_match(t, n)? {
                        out.insert(n.clone());
                        found = true;
                    }
                }
                if !found {
                    err.push_str(&format!(
                        "No external regressor labels matching regular expression '{t}' found.\n"
                    ));
                }
            } else if reg.names.iter().any(|n| n == t) {
                out.insert(t.to_string());
            } else {
                err.push_str(&format!(
                    "No external regressor matching '{t}' was found.\n"
                ));
            }
        }
        Ok(out.into_iter().collect())
    };
    let mut f_cols = std::collections::BTreeSet::new();
    for model in models.iter_mut() {
        for key in [
            "regress_ID",
            "info",
            "report",
            "detrend",
            "statistic",
            "regressors",
        ] {
            if model.get(key).is_none() {
                return Err(format!("external_regressor_config is missing '{key}'"));
            }
        }
        let statistic = model
            .get("statistic")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_lowercase();
        if statistic != "f" && statistic != "max_rp_corr" {
            return Err(format!(
                "external regressor statistic {statistic} is not valid (F or max_rp_corr)"
            ));
        }
        let full = expand(model.get("regressors").unwrap(), &mut err)?;
        // Every fit needs residual degrees of freedom (the F test divides by them).
        let fitted = if statistic == "f" { full.len() } else { 0 };
        if detrend_rank(model, n_vol).unwrap_or(1) + fitted >= n_vol {
            err.push_str(&format!(
                "{regress_id}: detrending and {fitted} regressors leave no residual degrees of freedom for {n_vol} timepoints.\n",
                regress_id = model.get("regress_ID").and_then(Json::as_str).unwrap_or("")
            ));
        }
        if let Some(Json::Obj(parts)) = model.get_mut("partial_models") {
            if statistic != "f" {
                return Err("partial_models need the F statistic".into());
            }
            for (_, names) in parts.iter_mut() {
                let expanded = expand(names, &mut err)?;
                *names = Json::strings(&expanded);
            }
        }
        let regress_id = model
            .get("regress_ID")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        if let Some(Json::Obj(parts)) = model.get("partial_models") {
            let mut extra = std::collections::BTreeSet::new();
            for (_, names) in parts {
                for n in names.as_arr().unwrap() {
                    let n = n.as_str().unwrap();
                    if !full.iter().any(|f| f == n) {
                        extra.insert(n.to_string());
                    }
                }
            }
            if !extra.is_empty() {
                err.push_str(&format!(
                    "Partial models in {regress_id} include regressors that are excluded from its full model: {:?}\n",
                    extra
                ));
            }
        }
        if statistic == "f" {
            f_cols.extend(full.clone());
        }
        *model.get_mut("regressors").unwrap() = Json::strings(&full);
    }
    for col in reg.cols.iter_mut() {
        for i in (0..col.len().saturating_sub(1)).rev() {
            if col[i].is_nan() {
                col[i] = col[i + 1];
            }
        }
        for i in 1..col.len() {
            if col[i].is_nan() {
                col[i] = col[i - 1];
            }
        }
    }
    // F models z-score their columns: a constant or non-finite one makes every statistic NaN
    // (tedana's scipy lstsq usually raises on it; float rounding can let a constant through).
    for n in &f_cols {
        let c = reg.col(n);
        if c.iter().any(|v| !v.is_finite()) || c.iter().all(|&v| v == c[0]) {
            err.push_str(&format!(
                "External regressor '{n}' is constant or not finite; an F model cannot use it.\n"
            ));
        }
    }
    let rows = reg.cols.first().map_or(0, Vec::len);
    if rows != n_vol {
        err.push_str(&format!(
            "External regressors have {rows} timepoints while fMRI data have {n_vol} timepoints.\n"
        ));
    }
    if err.is_empty() { Ok(()) } else { Err(err) }
}

/// `create_legendre_polynomial_basis_set(n, dtrank)`: P_0..P_{dtrank-1} on linspace(-1, 1, n).
fn legendre(n: usize, dtrank: usize) -> Mat {
    let step = 2.0 / (n as f64 - 1.0);
    let x: Vec<f64> = (0..n)
        .map(|i| {
            if i + 1 == n && n > 1 {
                1.0
            } else {
                i as f64 * step + -1.0
            }
        })
        .collect();
    let mut m = Mat::zeros(n, dtrank);
    for (t, &xt) in x.iter().enumerate() {
        let (mut p0, mut p1) = (1.0, xt);
        for v in 0..dtrank {
            m[(t, v)] = match v {
                0 => 1.0,
                1 => xt,
                _ => {
                    let p = ((2 * v - 1) as f64 * xt * p1 - (v - 1) as f64 * p0) / v as f64;
                    (p0, p1) = (p1, p);
                    p
                }
            };
        }
    }
    m
}

/// The detrending polynomials a config asks for: `true` is 1 + floor(n / 150), a positive
/// integer that many; None (the mean alone) otherwise.
fn detrend_rank(model: &Json, n: usize) -> Option<usize> {
    match model.get("detrend") {
        Some(Json::Bool(true)) => Some(1 + n / 150),
        Some(Json::Int(k)) if *k > 0 => Some((*k).min(n as i64) as usize),
        _ => None,
    }
}

/// Columns `base` then each regressor z-scored (scipy zscore of the contiguous series).
fn design(base: &Mat, reg: &Regressors, labels: &[String]) -> Mat {
    let mut x = Mat::zeros(base.rows, base.cols + labels.len());
    for c in 0..base.cols {
        x.col_mut(c).copy_from_slice(base.col(c));
    }
    for (j, l) in labels.iter().enumerate() {
        let col = x.col_mut(base.cols + j);
        col.copy_from_slice(reg.col(l));
        zscore(col);
    }
    x
}

/// `stats.fit_model`: per-column sum of squared residuals and the residual degrees of freedom.
fn sse(x: &Mat, y: &Mat) -> (Vec<f64>, f64) {
    let r = residual(x, y);
    let sse = (0..y.cols)
        .map(|c| r.col(c).iter().fold(0.0, |a, v| a + v * v))
        .collect();
    (sse, (y.rows - x.cols) as f64)
}

/// `fit_model_with_stats`: F, p (as 1 - cdf) and R2 of `full` over `base`.
fn f_test(base: &Mat, full: &Mat, y: &Mat) -> [Vec<f64>; 3] {
    let (sb, db) = sse(base, y);
    let (sf, df) = sse(full, y);
    let f: Vec<f64> = (0..y.cols)
        .map(|c| ((sb[c] - sf[c]) / (db - df)) / (sf[c] / df))
        .collect();
    let (d1, d2) = (db - df, df);
    let p = f
        .iter()
        .map(|&f| {
            let cdf = if f.is_nan() {
                f64::NAN
            } else if f <= 0.0 {
                0.0
            } else if f.is_infinite() {
                1.0
            } else {
                betainc(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))
            };
            1.0 - cdf
        })
        .collect();
    let r2 = (0..y.cols).map(|c| 1.0 - sf[c] / sb[c]).collect();
    [f, p, r2]
}

/// `fit_regressors`: the metric columns each config adds, in tedana's order. `seed` is the ICA
/// seed (tedana passes it to max_rp_corr's subsampling).
pub fn fit(config: &Json, reg: &Regressors, mixing: &Mat, seed: u32) -> Vec<(String, Vec<f64>)> {
    let mut out = Vec::new();
    for model in config.as_arr().unwrap_or_default() {
        let id = model.get("regress_ID").and_then(Json::as_str).unwrap_or("");
        let rank = detrend_rank(model, mixing.rows);
        let base = legendre(mixing.rows, rank.unwrap_or(1));
        let labels = model
            .get("regressors")
            .map(Json::str_list)
            .unwrap_or_default();
        if model
            .get("statistic")
            .and_then(Json::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("max_rp_corr"))
        {
            let mut rp = Mat::zeros(mixing.rows, labels.len());
            for (j, l) in labels.iter().enumerate() {
                rp.col_mut(j).copy_from_slice(reg.col(l));
            }
            let (m, rp) = if rank.is_some() {
                (residual(&base, mixing), residual(&base, &rp))
            } else {
                (mixing.clone(), rp)
            };
            out.push((
                format!("max_rp_corr {id} model"),
                max_rp_corr(&m, &rp, seed),
            ));
            continue;
        }
        let full = design(&base, reg, &labels);
        let [mut f, mut p, mut r2] =
            f_test(&base, &full, mixing).map(|v| vec![(format!("{id} model"), v)]);
        if let Some(Json::Obj(parts)) = model.get("partial_models") {
            for (name, drop) in parts {
                let drop = drop.str_list();
                let keep: Vec<String> = labels
                    .iter()
                    .filter(|l| !drop.contains(l))
                    .cloned()
                    .collect();
                let [pf, pp, pr] = f_test(&design(&base, reg, &keep), &full, mixing);
                let label = format!("{id} {name} partial model");
                f.push((label.clone(), pf));
                p.push((label.clone(), pp));
                r2.push((label, pr));
            }
        }
        for (stat, cols) in [("Fstat", f), ("pval", p), ("R2stat", r2)] {
            out.extend(cols.into_iter().map(|(n, v)| (format!("{stat} {n}"), v)));
        }
    }
    out
}

/// Column means and the centred columns' norms, as `np.corrcoef` scales its covariance.
fn centred(m: &Mat, rows: &[usize], f: impl Fn(f64) -> f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = rows.len() as f64;
    let cols: Vec<Vec<f64>> = (0..m.cols)
        .map(|c| {
            let v: Vec<f64> = rows.iter().map(|&r| f(m[(r, c)])).collect();
            let mean = crate::stats::sum(&v) / n;
            v.into_iter().map(|x| x - mean).collect()
        })
        .collect();
    let sd = cols
        .iter()
        .map(|v| (v.iter().fold(0.0, |a, x| a + x * x) / (n - 1.0)).sqrt())
        .collect();
    (cols, sd)
}

/// Pearson r of every `a` column with every `b` column (`utils.cross_correlation`), clipped.
fn cross_correlation(
    a: &(Vec<Vec<f64>>, Vec<f64>),
    b: &(Vec<Vec<f64>>, Vec<f64>),
    n: f64,
) -> Vec<Vec<f64>> {
    a.0.iter()
        .zip(&a.1)
        .map(|(x, sx)| {
            b.0.iter()
                .zip(&b.1)
                .map(|(y, sy)| {
                    let c = x.iter().zip(y).fold(0.0, |s, (p, q)| s + p * q) / (n - 1.0);
                    (c / sx / sy).clamp(-1.0, 1.0)
                })
                .collect()
        })
        .collect()
}

/// `calculate_max_rp_corr`: over 1000 random 90% subsamples of volumes, the mean of each
/// component's largest |r| with the 6N-column motion model (parameters, derivatives, both
/// shifted one volume either way), raw and squared.
fn max_rp_corr(mixing: &Mat, rp: &Mat, seed: u32) -> Vec<f64> {
    let (t, n) = (rp.rows, rp.cols);
    let mut rp2 = Mat::zeros(t, 2 * n);
    for j in 0..n {
        for r in 0..t {
            rp2[(r, j)] = rp[(r, j)];
            rp2[(r, n + j)] = if r == 0 {
                0.0
            } else {
                rp[(r, j)] - rp[(r - 1, j)]
            };
        }
    }
    let mut model = Mat::zeros(t, 6 * n);
    for j in 0..2 * n {
        for r in 0..t {
            model[(r, j)] = rp2[(r, j)];
            model[(r, 2 * n + j)] = if r == 0 { 0.0 } else { rp2[(r - 1, j)] };
            model[(r, 4 * n + j)] = if r + 1 == t { 0.0 } else { rp2[(r + 1, j)] };
        }
    }
    let k = (0.9 * t as f64).round_ties_even() as usize;
    let mut rng = Pcg64::new(seed as u64);
    let splits: Vec<Vec<usize>> = (0..1000).map(|_| rng.choice(t, k)).collect();
    let per_split = |rows: &Vec<usize>| -> Vec<f64> {
        let sq = |v: f64| v * v;
        let raw = cross_correlation(
            &centred(mixing, rows, |v| v),
            &centred(&model, rows, |v| v),
            k as f64,
        );
        let squared = cross_correlation(
            &centred(mixing, rows, sq),
            &centred(&model, rows, sq),
            k as f64,
        );
        (0..mixing.cols)
            .map(|c| {
                squared[c]
                    .iter()
                    .chain(&raw[c])
                    .filter(|v| !v.is_nan())
                    .fold(f64::NAN, |m, v| {
                        if m.is_nan() || v.abs() > m {
                            v.abs()
                        } else {
                            m
                        }
                    })
            })
            .collect()
    };
    #[cfg(feature = "parallel")]
    let maxima: Vec<Vec<f64>> = {
        use rayon::prelude::*;
        splits.par_iter().map(per_split).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let maxima: Vec<Vec<f64>> = splits.iter().map(per_split).collect();
    (0..mixing.cols)
        .map(|c| {
            let v: Vec<f64> = maxima
                .iter()
                .map(|m| m[c])
                .filter(|v| !v.is_nan())
                .collect();
            crate::stats::sum(&v) / v.len() as f64
        })
        .collect()
}

const F_HP: f64 = 0.01;

/// HFC's frequency step and the FFT bins above F_HP. Empty (tedana fails on it too) when the TR
/// is not in seconds or the series is too short.
pub fn hfc_bins(n: usize, tr: f32) -> (f64, Vec<usize>) {
    let step = (1.0 / (n as f32 * tr)) as f64;
    (
        step,
        (1..=n / 2).filter(|&k| k as f64 * step > F_HP).collect(),
    )
}

/// `calculate_hfc`: the normalized frequency above 0.01 Hz where each component's cumulative
/// power first comes closest to half its total. tedana's TR is the header's float32, and numpy 2
/// keeps the frequency step and the Nyquist band in float32.
pub fn hfc(mixing: &Mat, tr: f32) -> Vec<f64> {
    let n = mixing.rows;
    let (step, bins) = hfc_bins(n, tr);
    let band = (1.0 / tr / 2.0 - F_HP as f32) as f64;
    (0..mixing.cols)
        .map(|c| {
            let (mut re, mut im) = (mixing.col(c).to_vec(), vec![0.0; n]);
            fftn(&mut re, &mut im, &[n]);
            let power: Vec<f64> = bins
                .iter()
                .map(|&k| re[k] * re[k] + im[k] * im[k])
                .collect();
            let total = power.iter().fold(0.0, |a, p| a + p);
            let mut cum = 0.0;
            let mut best = (f64::INFINITY, 0);
            for (i, p) in power.iter().enumerate() {
                cum += p;
                let d = (cum / total - 0.5).abs();
                if d < best.0 {
                    best = (d, i);
                }
            }
            (bins[best.1] as f64 * step - F_HP) / band
        })
        .collect()
}

/// tedana's "external regressor correlation" columns: `np.corrcoef` of every regressor with
/// every (sign-flipped) mixing column, regressors in file order.
pub fn correlations(reg: &Regressors, mixing: &Mat) -> Vec<(String, Vec<f64>)> {
    let t = mixing.rows;
    let rows: Vec<usize> = (0..t).collect();
    let mut r = Mat::zeros(t, reg.cols.len());
    for (j, col) in reg.cols.iter().enumerate() {
        r.col_mut(j).copy_from_slice(col);
    }
    let corr = cross_correlation(
        &centred(&r, &rows, |v| v),
        &centred(mixing, &rows, |v| v),
        t as f64,
    );
    reg.names
        .iter()
        .zip(corr)
        .map(|(n, v)| (format!("external regressor correlation {n}"), v))
        .collect()
}

/// Python `re.match(pattern, text, re.IGNORECASE)` for the regex subset decision trees use:
/// literals, `.`, classes, groups, `(?:` `(?=` `(?!`, `|`, `* + ? {m,n}` (lazy too), `^ $`.
pub fn py_match(pattern: &str, text: &str) -> Result<bool, String> {
    let p: Vec<char> = pattern.chars().collect();
    let mut pos = 0;
    let node = parse_alt(&p, &mut pos).map_err(|e| format!("regex '{pattern}': {e}"))?;
    if pos != p.len() {
        return Err(format!("regex '{pattern}': unbalanced ')'"));
    }
    let t: Vec<char> = text.chars().flat_map(char::to_lowercase).collect();
    Ok(m(&node, &t, 0, &mut |_| true))
}

enum Re {
    Char(char),
    Any,
    Class(Vec<(char, char)>, bool),
    Start,
    End,
    Seq(Vec<Re>),
    Alt(Vec<Re>),
    Look(Box<Re>, bool),
    Repeat(Box<Re>, usize, usize, bool),
}

fn parse_alt(p: &[char], i: &mut usize) -> Result<Re, String> {
    let mut alts = vec![parse_seq(p, i)?];
    while *i < p.len() && p[*i] == '|' {
        *i += 1;
        alts.push(parse_seq(p, i)?);
    }
    Ok(if alts.len() == 1 {
        alts.pop().unwrap()
    } else {
        Re::Alt(alts)
    })
}

fn parse_seq(p: &[char], i: &mut usize) -> Result<Re, String> {
    let mut items = Vec::new();
    while *i < p.len() && p[*i] != '|' && p[*i] != ')' {
        let atom = parse_atom(p, i)?;
        items.push(parse_quantifier(p, i, atom)?);
    }
    Ok(Re::Seq(items))
}

fn lower(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

fn escape(c: char) -> Result<Re, String> {
    let class = |r: &[(char, char)], neg| Re::Class(r.to_vec(), neg);
    let digit = [('0', '9')];
    let word = [('a', 'z'), ('0', '9'), ('_', '_')];
    let space = [(' ', ' '), ('\t', '\r')];
    Ok(match c {
        'd' => class(&digit, false),
        'D' => class(&digit, true),
        'w' => class(&word, false),
        'W' => class(&word, true),
        's' => class(&space, false),
        'S' => class(&space, true),
        c if c.is_ascii_alphanumeric() => return Err(format!("\\{c} is not supported")),
        c => Re::Char(lower(c)),
    })
}

fn parse_atom(p: &[char], i: &mut usize) -> Result<Re, String> {
    let c = p[*i];
    *i += 1;
    Ok(match c {
        '.' => Re::Any,
        '^' => Re::Start,
        '$' => Re::End,
        '\\' => {
            let e = *p.get(*i).ok_or("trailing backslash")?;
            *i += 1;
            escape(e)?
        }
        '[' => {
            let neg = p.get(*i) == Some(&'^');
            if neg {
                *i += 1;
            }
            let mut ranges = Vec::new();
            let mut first = true;
            while *i < p.len() && (p[*i] != ']' || first) {
                first = false;
                let mut a = p[*i];
                *i += 1;
                if a == '\\' {
                    match escape(*p.get(*i).ok_or("trailing backslash")?)? {
                        Re::Char(c) => a = c,
                        Re::Class(r, false) => {
                            *i += 1;
                            ranges.extend(r);
                            continue;
                        }
                        _ => return Err("negated escapes inside [] are not supported".into()),
                    }
                    *i += 1;
                }
                if p.get(*i) == Some(&'-') && p.get(*i + 1).is_some_and(|&b| b != ']') {
                    let b = p[*i + 1];
                    *i += 2;
                    ranges.push((lower(a), lower(b)));
                    ranges.push((a.to_ascii_uppercase(), b.to_ascii_uppercase()));
                } else {
                    ranges.push((lower(a), lower(a)));
                }
            }
            if *i >= p.len() {
                return Err("unterminated [".into());
            }
            *i += 1;
            Re::Class(ranges, neg)
        }
        '(' => {
            let kind = if p.get(*i) == Some(&'?') {
                let k = *p.get(*i + 1).ok_or("unterminated group")?;
                *i += 2;
                match k {
                    ':' => 0,
                    '=' => 1,
                    '!' => 2,
                    _ => return Err(format!("(?{k} is not supported")),
                }
            } else {
                0
            };
            let inner = parse_alt(p, i)?;
            if p.get(*i) != Some(&')') {
                return Err("unterminated group".into());
            }
            *i += 1;
            match kind {
                0 => inner,
                k => Re::Look(Box::new(inner), k == 2),
            }
        }
        '*' | '+' | '?' | '{' => return Err(format!("nothing to repeat before '{c}'")),
        c => Re::Char(lower(c)),
    })
}

fn parse_quantifier(p: &[char], i: &mut usize, atom: Re) -> Result<Re, String> {
    let (lo, hi) = match p.get(*i) {
        Some('*') => (0, usize::MAX),
        Some('+') => (1, usize::MAX),
        Some('?') => (0, 1),
        Some('{') => {
            let close = p[*i..]
                .iter()
                .position(|&c| c == '}')
                .ok_or("unterminated {")?
                + *i;
            let body: String = p[*i + 1..close].iter().collect();
            let num = |s: &str| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad repeat {{{body}}}"))
            };
            let (lo, hi) = match body.split_once(',') {
                None => (num(&body)?, num(&body)?),
                Some((a, "")) => (if a.is_empty() { 0 } else { num(a)? }, usize::MAX),
                Some((a, b)) => (if a.is_empty() { 0 } else { num(a)? }, num(b)?),
            };
            *i = close;
            (lo, hi)
        }
        _ => return Ok(atom),
    };
    *i += 1;
    let lazy = p.get(*i) == Some(&'?');
    if lazy {
        *i += 1;
    }
    Ok(Re::Repeat(Box::new(atom), lo, hi, lazy))
}

/// Backtracking match of `r` at `i`, then the continuation `k`.
fn m(r: &Re, t: &[char], i: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match r {
        Re::Char(c) => i < t.len() && t[i] == *c && k(i + 1),
        Re::Any => i < t.len() && t[i] != '\n' && k(i + 1),
        Re::Class(ranges, neg) => {
            i < t.len() && ranges.iter().any(|&(a, b)| a <= t[i] && t[i] <= b) != *neg && k(i + 1)
        }
        Re::Start => i == 0 && k(i),
        Re::End => (i == t.len() || (i + 1 == t.len() && t[i] == '\n')) && k(i),
        Re::Seq(items) => seq(items, t, i, k),
        Re::Alt(alts) => alts.iter().any(|a| m(a, t, i, k)),
        Re::Look(inner, negative) => m(inner, t, i, &mut |_| true) != *negative && k(i),
        Re::Repeat(inner, lo, hi, lazy) => repeat(inner, *lo, *hi, *lazy, t, i, 0, k),
    }
}

fn seq(items: &[Re], t: &[char], i: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match items.split_first() {
        None => k(i),
        Some((first, rest)) => m(first, t, i, &mut |j| seq(rest, t, j, k)),
    }
}

#[allow(clippy::too_many_arguments)]
fn repeat(
    r: &Re,
    lo: usize,
    hi: usize,
    lazy: bool,
    t: &[char],
    i: usize,
    n: usize,
    k: &mut dyn FnMut(usize) -> bool,
) -> bool {
    let more = |k: &mut dyn FnMut(usize) -> bool| {
        n < hi
            && m(r, t, i, &mut |j| {
                j != i && repeat(r, lo, hi, lazy, t, j, n + 1, k)
            })
    };
    if n < lo {
        return n < hi && m(r, t, i, &mut |j| repeat(r, lo, hi, lazy, t, j, n + 1, k));
    }
    if lazy {
        k(i) || more(k)
    } else {
        more(k) || k(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_regex_subset() {
        for (p, s, want) in [
            ("^mot_.*$", "Mot_X", true),
            ("^mot_.*$", "rot_x", false),
            ("^(?!signal).*$", "signal_left", false),
            ("^(?!signal).*$", "csf", true),
            ("^.*$", "anything", true),
            ("^csf.*$", "CSF_wm", true),
            ("^(trans|rot)_[xyz]$", "rot_Y", true),
            ("^(trans|rot)_[xyz]$", "rot_yaw", false),
            ("^trans_x(_derivative1)?(_power2)?$", "trans_x_power2", true),
            ("^a{2,3}b", "aaab_c", true),
            ("^a{2,3}?b", "ab", false),
            ("^\\w+\\d$", "mot_3", true),
        ] {
            assert_eq!(py_match(p, s).unwrap(), want, "{p} on {s}");
        }
        assert!(py_match("^(?<name>x)", "x").is_err());
    }

    #[test]
    fn legendre_and_linspace_match_numpy() {
        let m = legendre(5, 3);
        assert_eq!(m.col(1), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(m.col(2), &[1.0, -0.125, -0.5, -0.125, 1.0]);
    }

    #[test]
    fn rank_deficient_f_test_matches_tedana() {
        // Two identical regressors: tedana (numpy lstsq, df from the column count) gives
        // F = 1, p = scipy f.sf(1, 2, 1), R2 = 2/3.
        let base = legendre(4, 1);
        let mut full = Mat::zeros(4, 3);
        for (c, v) in [[1.0; 4], [-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]]
            .iter()
            .enumerate()
        {
            full.col_mut(c).copy_from_slice(v);
        }
        let y = Mat::from_row_major(4, 1, &[-2.0, 0.0, 1.0, 1.0]);
        let [f, p, r2] = f_test(&base, &full, &y);
        assert!((f[0] - 1.0).abs() < 1e-12);
        assert!((p[0] - 0.5773502691896257).abs() < 1e-12);
        assert!((r2[0] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn unusable_f_columns_are_rejected() {
        let config = r#"[{"regress_ID": "m", "info": "", "report": "", "detrend": false,
            "statistic": "F", "regressors": ["x"]}]"#;
        let check = |col: &str| {
            let mut reg = Regressors::parse(&format!("x\tok\n{col}")).unwrap();
            validate(&mut reg, &mut Json::parse(config).unwrap(), 4)
        };
        for bad in [
            "1\t1\n1\t2\n1\t3\n1\t4",
            "n/a\t1\nn/a\t2\nn/a\t3\nn/a\t4",
            "1\t1\n2\t2\ninf\t3\n4\t4",
        ] {
            assert!(check(bad).unwrap_err().contains("'x'"), "{bad}");
        }
        assert!(check("1\t1\nn/a\t2\n3\t3\n4\t4").is_ok());
    }
}
