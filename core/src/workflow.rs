//! tedana's default workflow end to end, writing its BIDS-named outputs (minus the HTML report,
//! figures, report.txt and references.bib).

use crate::decay::fit_echoes;
use crate::external;
use crate::fastica::f_ica;
use crate::io::{Nifti, parse_nifti};
use crate::json::Json;
use crate::linalg::{Mat, residual};
use crate::mapca::{Criterion, MaPca};
use crate::math::exp;
use crate::metrics::{Metrics, coeffs, row_coeffs};
use crate::output::{Column, Files, Voxels, Writer, component_name};
use crate::pipeline::{
    Combined, check_te_values, combine, component_metrics, component_table, metric_columns, tedpca,
};
use crate::selection::{Selector, load_tree};
use crate::stats::{demean, median, sum, sum_f32, zscore, zscore_columns};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const METADATA: &str = include_str!("../trees/metric_metadata.json");

pub struct Options {
    /// Echo times as given: all seconds (< 1) or all milliseconds.
    pub tes: Vec<f64>,
    /// Bundled tree name or a tree file's contents.
    pub tree: String,
    pub criterion: Criterion,
    pub seed: u32,
    pub maxit: usize,
    pub maxrestart: usize,
    /// tedana --prefix: output names start with "{prefix}_".
    pub prefix: String,
    pub command: String,
    /// tedana --external: the regressors TSV's contents, for trees with external_regressor_config.
    pub external: Option<String>,
    /// tedana --no-reports, which also drops the metrics' "external regressor correlation" columns.
    pub no_reports: bool,
    /// tedana --mask: a NIfTI file's bytes on the echoes' grid; voxels > 0 replace the EPI mask.
    pub mask: Option<Vec<u8>>,
    /// tedana --dummy-scans: volumes dropped from the start of every echo (and of the regressors
    /// when they still include them).
    pub dummy_scans: usize,
}

impl Default for Options {
    /// tedana's defaults.
    fn default() -> Self {
        Options {
            tes: Vec::new(),
            tree: "tedana_orig".into(),
            criterion: Criterion::Aic,
            seed: 42,
            maxit: 500,
            maxrestart: 10,
            prefix: String::new(),
            command: String::new(),
            external: None,
            no_reports: false,
            mask: None,
            dummy_scans: 0,
        }
    }
}

pub struct Summary {
    pub n_components: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub seed: u32,
    pub n_iter: usize,
}

fn obj(kv: Vec<(&str, Json)>) -> Json {
    Json::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn floats(v: &[f64]) -> Json {
    Json::Arr(v.iter().map(|&x| Json::Num(x)).collect())
}

/// tedana's output file name for `base` under `--prefix`.
pub fn output_name(prefix: &str, base: &str) -> String {
    if prefix.is_empty() {
        base.to_string()
    } else {
        format!("{}_{base}", prefix.trim_end_matches('_'))
    }
}

/// The leading "Component" column of a component table.
fn component_column(prefix: &str, n: usize) -> (String, Column) {
    let names = (0..n).map(|k| component_name(prefix, k, n)).collect();
    ("Component".to_string(), Column::Text(names))
}

/// A decomposition json: the method, then each component's description.
fn decomposition(method: &str, names: impl Iterator<Item = String>, description: &str) -> Json {
    let mut kv = vec![("Method".to_string(), Json::from(method))];
    kv.extend(names.map(|n| {
        let entry = obj(vec![
            ("Description", description.into()),
            ("Method", "tedana".into()),
        ]);
        (n, entry)
    }));
    Json::Obj(kv)
}

/// numpy's linear-method percentile of float32 values, interpolated in float64 from a float32
/// difference (numpy `_lerp`).
fn percentile_linear_f32(sorted: &[f32], q: f64) -> f64 {
    let q = q / 100.0;
    let (alpha, beta) = (1.0, 1.0); // numpy's _compute_virtual_index for the linear method
    let index = sorted.len() as f64 * q + (alpha + q * (1.0 - alpha - beta)) - 1.0;
    let lo = (index.floor().max(0.0) as usize).min(sorted.len() - 1);
    let hi = (lo + 1).min(sorted.len() - 1);
    let gamma = index - index.floor();
    let (a, b) = (sorted[lo], sorted[hi]);
    let diff = (b - a) as f64;
    if gamma >= 0.5 {
        b as f64 - diff * (1.0 - gamma)
    } else {
        a as f64 + diff * gamma
    }
}

/// tedana `rmse_of_fit_decay_ts(fitmode="all")` in float32: per-voxel map and the confounds table.
fn decay_rmse(c: &Combined, tes: &[f64], n_vol: usize) -> (Vec<f32>, Vec<(String, Column)>) {
    let n_echo = tes.len();
    let n_vox = c.voxels.len();
    let mut rmse = vec![f32::NAN; n_vox * n_vol];
    for v in (0..n_vox).filter(|&v| c.masksum[v] > 0) {
        let k = fit_echoes(c.masksum[v]);
        let predicted: Vec<f32> = tes[..k]
            .iter()
            .map(|te| (c.s0[v] * exp(-te / c.t2s[v])) as f32)
            .collect();
        for t in 0..n_vol {
            let ss = (0..k).fold(-0.0f32, |a, e| {
                let d = c.data[(v * n_echo + e) * n_vol + t] - predicted[e];
                a + d * d
            });
            rmse[v * n_vol + t] = ((ss as f64 / k as f64) as f32).sqrt();
        }
    }
    let map: Vec<f32> = (0..n_vox)
        .map(|v| {
            let row: Vec<f32> = rmse[v * n_vol..(v + 1) * n_vol]
                .iter()
                .map(|x| if x.is_nan() { 0.0 } else { *x })
                .collect();
            let count = rmse[v * n_vol..(v + 1) * n_vol]
                .iter()
                .filter(|x| !x.is_nan())
                .count();
            if count == 0 {
                f32::NAN
            } else {
                (sum_f32(&row) as f64 / count as f64) as f32
            }
        })
        .collect();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 9];
    for t in 0..n_vol {
        let vals: Vec<f32> = (0..n_vox)
            .map(|v| rmse[v * n_vol + t])
            .filter(|x| !x.is_nan())
            .collect();
        let n = vals.len() as f64;
        let mean = (vals.iter().fold(-0.0f32, |a, b| a + b) as f64 / n) as f32;
        let var = (vals
            .iter()
            .fold(-0.0f32, |a, x| a + (x - mean) * (x - mean)) as f64
            / n) as f32;
        let mut sorted = vals.clone();
        sorted.sort_by(f32::total_cmp);
        columns[0].push(mean as f64);
        columns[1].push(var.sqrt() as f64);
        for (i, q) in [0.0, 2.0, 25.0, 50.0, 75.0, 98.0, 100.0].iter().enumerate() {
            columns[2 + i].push(percentile_linear_f32(&sorted, *q));
        }
    }
    let names = [
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_percentile02",
        "rmse_percentile25",
        "rmse_median",
        "rmse_percentile75",
        "rmse_percentile98",
        "rmse_max",
    ];
    (
        map,
        names
            .iter()
            .zip(columns)
            .map(|(n, c)| (n.to_string(), Column::Float(c)))
            .collect(),
    )
}

/// tedana `generate_decay_metrics`.
fn decay_metrics(c: &Combined, rmse_map: &[f32]) -> Json {
    let fit: Vec<usize> = (0..c.voxels.len()).filter(|&v| c.masksum[v] >= 1).collect();
    // T2* and S0 skip zeros; RMSE keeps them.
    let mean_median = |x: &dyn Fn(usize) -> f64, skip_zero: bool| {
        let v: Vec<f64> = fit
            .iter()
            .map(|&i| x(i))
            .filter(|x| x.is_finite() && !(skip_zero && *x == 0.0))
            .collect();
        if v.is_empty() {
            (None, None)
        } else {
            (Some(sum(&v) / v.len() as f64), Some(median(&v)))
        }
    };
    let (t2_mean, t2_median) = mean_median(&|i| c.t2s[i], true);
    let (s0_mean, s0_median) = mean_median(&|i| c.s0[i], true);
    let (rmse_mean, rmse_median) = mean_median(&|i| rmse_map[i] as f64, false);
    let counts: Vec<(String, Json)> = (1..=c.masksum.iter().copied().max().unwrap_or(0))
        .map(|n| (n, fit.iter().filter(|&&v| c.masksum[v] == n).count()))
        .filter(|(_, k)| *k > 0)
        .map(|(n, k)| (n.to_string(), Json::Int(k as i64)))
        .collect();
    obj(vec![
        ("t2star_mean", t2_mean.into()),
        ("t2star_median", t2_median.into()),
        ("s0_mean", s0_mean.into()),
        ("s0_median", s0_median.into()),
        ("rmse_mean", rmse_mean.into()),
        ("rmse_median", rmse_median.into()),
        ("n_voxels_base_mask", Json::Int(c.voxels.len() as i64)),
        ("n_voxels_fit_mask", Json::Int(fit.len() as i64)),
        ("good_echo_voxel_counts", Json::Obj(counts)),
    ])
}

/// tedana `get_metadata` for the columns written.
fn metadata(columns: &[&str]) -> Json {
    let all = Json::parse(METADATA).unwrap();
    let Json::Obj(kv) = all else { unreachable!() };
    Json::Obj(
        kv.into_iter()
            .filter(|(k, _)| k == "Component" || columns.contains(&k.as_str()))
            .collect(),
    )
}

/// Metric columns shared by the PCA and ICA tables; counts and signs print as integers.
fn metric_tsv(
    m: &Metrics,
    normalized: &[f64],
    with_noise: bool,
    with_dtable: bool,
) -> Vec<(String, Column)> {
    metric_columns(m)
        .into_iter()
        .filter(|(n, _)| {
            (with_noise || *n != "countnoise") && (with_dtable || *n != "d_table_score")
        })
        .map(|(n, v)| {
            let col = match n {
                "normalized variance explained" => Column::Float(normalized.to_vec()),
                "countsigFT2" | "countsigFS0" | "countnoise" | "optimal sign" => {
                    Column::Int(v.iter().map(|&x| x as i64).collect())
                }
                _ => Column::Float(v),
            };
            (n.to_string(), col)
        })
        .collect()
}

fn mixing_columns(prefix: &str, mixing: &Mat, max_value: usize) -> Vec<(String, Column)> {
    (0..mixing.cols)
        .map(|k| {
            (
                component_name(prefix, k, max_value),
                Column::Float(mixing.col(k).to_vec()),
            )
        })
        .collect()
}

/// Per-voxel rows of a subset (positions in the grid) to a NIfTI-ordered float32 volume series.
fn grid(
    n_grid: usize,
    positions: &[usize],
    n_t: usize,
    value: impl Fn(usize, usize) -> f64,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_grid * n_t];
    for (r, &i) in positions.iter().enumerate() {
        for t in 0..n_t {
            out[i + n_grid * t] = value(r, t) as f32;
        }
    }
    out
}

fn mapca_json(m: &MaPca) -> Json {
    let total =
        |n: usize| Json::Num((m.cumulative_varex[n - 1] * 1000.0).round_ties_even() / 1000.0);
    let curve = |n: usize, c: &[f64]| {
        obj(vec![
            ("n_components", Json::Int(n as i64)),
            ("explained_variance_total", total(n)),
            ("curve", floats(c)),
        ])
    };
    let count = |n: usize| {
        obj(vec![
            ("n_components", Json::Int(n as i64)),
            ("explained_variance_total", total(n)),
        ])
    };
    obj(vec![
        ("aic", curve(m.aic.0, &m.aic.1)),
        ("kic", curve(m.kic.0, &m.kic.1)),
        ("mdl", curve(m.mdl.0, &m.mdl.1)),
        ("varex_90", count(m.varex_90)),
        ("varex_95", count(m.varex_95)),
        (
            "MAPCA_subsampling",
            obj(vec![
                (
                    "calculated_IID_subsample_depth",
                    Json::Int(m.subsample_depth as i64),
                ),
                ("calculated_IID_subsample_mean", Json::Num(m.iid_mean)),
                ("effective_num_IID_samples", Json::Num(m.effective_samples)),
                ("total_num_samples", Json::Int(m.u.rows as i64)),
            ]),
        ),
    ])
}

/// Load and validate `--external` against the tree before any processing, as tedana does; the
/// tree keeps the expanded regressor lists (they are written with it).
fn external_regressors(
    tree: &mut Json,
    tsv: Option<&str>,
    n_vol: usize,
    dummy_scans: usize,
) -> Result<Option<external::Regressors>, String> {
    let config = tree
        .get_mut("external_regressor_config")
        .filter(|v| **v != Json::Null);
    match (config, tsv) {
        (None, None) => Ok(None),
        (None, Some(_)) => {
            Err("--external needs a decision tree with external_regressor_config".into())
        }
        (Some(_), None) => Err("this decision tree needs --external regressors".into()),
        (Some(config), Some(tsv)) => {
            let mut reg = external::Regressors::parse(tsv)?;
            external::validate(&mut reg, config, n_vol, dummy_scans)?;
            Ok(Some(reg))
        }
    }
}

/// tedana `load_mask`: voxels > 0 of a 3D mask (or 4D with one volume) on the echoes' grid. Like
/// tedana, only the shape is checked, not the affine.
fn brain_mask(bytes: &[u8], dims: [usize; 4]) -> Result<Vec<bool>, String> {
    let m = parse_nifti(bytes.to_vec()).map_err(|e| format!("mask: {e}"))?;
    if m.dims[..3] != dims[..3] || m.dims[3] != 1 {
        return Err(format!(
            "mask is {:?}; the echoes' grid is {:?}",
            m.dims,
            &dims[..3]
        ));
    }
    Ok((0..dims[..3].iter().product())
        .map(|i| m.value(i) > 0.0)
        .collect())
}

fn tree_needs(tree: &Json, metric: &str) -> bool {
    tree.get("necessary_metrics")
        .and_then(Json::as_arr)
        .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(metric)))
}

/// Run tedana's default workflow on the echoes. `input_names` are what the registry lists for
/// them; files go to `w` (a directory, or memory in the browser) and are returned when kept.
pub fn run(
    mut echoes: Vec<Nifti>,
    input_names: &[String],
    mut w: Writer,
    opts: &Options,
    log: &mut dyn FnMut(&str),
) -> Result<(Summary, Files), String> {
    let tes = check_te_values(&opts.tes)?;
    let mut tree = load_tree(&opts.tree)?;
    for e in &mut echoes {
        e.drop_volumes(opts.dummy_scans)?;
    }
    let n_vol = echoes.first().map_or(0, |e| e.dims[3]);
    let external =
        external_regressors(&mut tree, opts.external.as_deref(), n_vol, opts.dummy_scans)?;
    let user_mask = match (&opts.mask, echoes.first()) {
        (Some(bytes), Some(e)) => Some(brain_mask(bytes, e.dims)?),
        _ => None,
    };
    let tr = echoes.first().map_or(0.0, |e| e.header.pixdim[4]);
    if tree_needs(&tree, "HFC") && external::hfc_bins(n_vol, tr).1.is_empty() {
        return Err(format!(
            "HFC needs frequencies above 0.01 Hz; TR {tr} s with {n_vol} volumes has none"
        ));
    }
    // Output names join the directory, so a prefix must not be a path.
    if opts.prefix.contains(['/', '\\']) || opts.prefix.contains("..") {
        return Err(format!("prefix '{}' must not contain a path", opts.prefix));
    }
    log("Computing EPI mask, adaptive mask, T2*/S0 maps and optimal combination");
    let mut c = combine(&echoes, &tes, user_mask)?;
    let (dims, n_vol) = (
        [echoes[0].dims[0], echoes[0].dims[1], echoes[0].dims[2]],
        echoes[0].dims[3],
    );
    let n_grid = dims.iter().product();
    let n_echo = tes.len();
    let name = |base: &str| output_name(&opts.prefix, base);

    w.set_template(echoes.into_iter().next().unwrap().header);
    let epi = &c.voxels;
    let mut mask_img = vec![0i32; n_grid];
    for (v, &i) in epi.iter().enumerate() {
        mask_img[i] = c.masksum[v];
    }
    w.image(
        "adaptive mask img",
        &name("desc-adaptiveGoodSignal_mask.nii.gz"),
        &dims,
        Voxels::I32(&mask_img),
    )?;
    // tedana writes component maps 4D even when there is one.
    let d4 = &[dims[0], dims[1], dims[2], n_vol];
    let d3 = &dims;
    w.image(
        "t2star img",
        &name("T2starmap.nii.gz"),
        d3,
        Voxels::F32(&grid(n_grid, epi, 1, |v, _| c.t2s[v])),
    )?;
    w.image(
        "s0 img",
        &name("S0map.nii.gz"),
        d3,
        Voxels::F32(&grid(n_grid, epi, 1, |v, _| c.s0[v])),
    )?;
    let (rmse_map, confounds) = decay_rmse(&c, &tes, n_vol);
    w.image(
        "rmse img",
        &name("desc-rmse_statmap.nii.gz"),
        d3,
        Voxels::F32(&grid(n_grid, epi, 1, |v, _| rmse_map[v] as f64)),
    )?;
    w.tsv(
        "confounds tsv",
        &name("desc-confounds_timeseries.tsv"),
        &confounds,
    )?;
    w.json(
        "decay metrics json",
        &name("desc-decay_metrics.json"),
        &decay_metrics(&c, &rmse_map),
    )?;
    w.image(
        "combined img",
        &name("desc-optcom_bold.nii.gz"),
        d4,
        Voxels::F32(&grid(n_grid, epi, n_vol, |v, t| c.optcom[v * n_vol + t])),
    )?;

    log("Computing PCA of the optimally combined data (maPCA)");
    let r = tedpca(&c, dims, opts.criterion)?;
    log("Computing PCA component metrics");
    let n_comp = r.mapca.explained_variance.len();
    // Base-mask voxels with at least `min` good echoes: grid positions and optcom series.
    let subset = |min: i32| {
        let rows: Vec<usize> = (0..epi.len()).filter(|&v| c.masksum[v] >= min).collect();
        let pos: Vec<usize> = rows.iter().map(|&v| epi[v]).collect();
        let optcom: Vec<f64> = rows
            .iter()
            .flat_map(|&v| c.optcom[v * n_vol..(v + 1) * n_vol].iter().copied())
            .collect();
        (pos, optcom)
    };
    let (clf_pos, clf_optcom) = subset(3);
    let (pm, comp_ts) = component_metrics(&c, dims, &tes, &r.mapca.components.t());
    let ratio: Vec<f64> = r
        .mapca
        .explained_variance_ratio
        .iter()
        .map(|v| v * 100.0)
        .collect();
    let mut pca_cols = vec![component_column("PCA", n_comp)];
    pca_cols.extend(metric_tsv(&pm, &ratio, true, true));
    pca_cols.push((
        "estimated normalized variance explained".into(),
        Column::Float(pm.normalized_variance_explained.clone()),
    ));
    pca_cols.push((
        "classification".into(),
        Column::Text(vec!["accepted".into(); n_comp]),
    ));
    pca_cols.push((
        "rationale".into(),
        Column::Text(vec![String::new(); n_comp]),
    ));
    let pca_maps = row_coeffs(&clf_optcom, n_vol, &zscore_columns(&comp_ts), |y| zscore(y));
    w.image(
        "z-scored PCA components img",
        &name("desc-PCA_stat-z_components.nii.gz"),
        &[dims[0], dims[1], dims[2], n_comp],
        Voxels::F32(&grid(n_grid, &clf_pos, n_comp, |r, k| pca_maps[(r, k)])),
    )?;
    w.tsv(
        "PCA mixing tsv",
        &name("desc-PCA_mixing.tsv"),
        &mixing_columns("pca", &comp_ts, n_comp - 1),
    )?;
    let pca_names: Vec<&str> = pca_cols.iter().map(|(n, _)| n.as_str()).collect();
    w.json(
        "PCA metrics json",
        &name("desc-PCA_metrics.json"),
        &metadata(&pca_names),
    )?;
    w.tsv("PCA metrics tsv", &name("desc-PCA_metrics.tsv"), &pca_cols)?;
    let decomp = decomposition(
        "Principal components analysis implemented by sklearn. Components are sorted by variance explained in descending order. ",
        (0..n_comp).map(|k| component_name("pca", k, n_comp - 1)),
        "PCA fit to optimally combined data.",
    );
    w.json(
        "PCA decomposition json",
        &name("desc-PCA_decomposition.json"),
        &decomp,
    )?;
    w.json(
        "PCA cross component metrics json",
        &name("desc-PCACrossComponent_metrics.json"),
        &mapca_json(&r.mapca),
    )?;
    log(&format!(
        "Selected {n_comp} components ({:?})",
        opts.criterion
    ));

    // ICA, metrics and selection; restart from the next seed when no component looks like BOLD.
    let (mut seed, mut n_restarts) = (opts.seed, 0usize);
    let (selector, mixing, metrics, ica_seed, n_iter) = loop {
        let ica = f_ica(
            &r.kept_data,
            n_comp,
            seed,
            opts.maxit,
            opts.maxrestart.saturating_sub(n_restarts),
        );
        log(&format!(
            "ICA with random seed {} converged {} in {} iterations",
            ica.seed, ica.fit.converged, ica.fit.n_iter
        ));
        seed = ica.seed + 1;
        n_restarts = (seed - opts.seed) as usize;
        let (m, mixing) = component_metrics(&c, dims, &tes, &ica.mixing);
        let mut table = component_table(&m);
        if tree_needs(&tree, "HFC") {
            table
                .columns
                .push(("HFC".into(), external::hfc(&mixing, tr)));
        }
        if let Some(reg) = &external {
            log("Fitting external regressors");
            let config = tree.get("external_regressor_config").unwrap();
            table
                .columns
                .extend(external::fit(config, reg, &mixing, ica.seed));
        }
        log("Selecting components");
        let s = Selector::select(tree.clone(), table, n_echo, n_vol)?;
        let likely_bold = s
            .table
            .tags
            .iter()
            .filter(|t| t.iter().any(|x| x.contains("Likely BOLD")))
            .count();
        if likely_bold == 0 && n_restarts < opts.maxrestart {
            log("No BOLD components found. Re-attempting ICA.");
            continue;
        }
        break (s, mixing, m, ica.seed, ica.fit.n_iter);
    };
    // The per-echo data and PCA inputs are done; free them before the 4D output writes, the peak.
    c.data = Vec::new();
    drop(r);
    let mut s = selector;
    let class = s.table.classification.clone();
    let acc: Vec<usize> = (0..n_comp).filter(|&k| class[k] == "accepted").collect();
    let rej: Vec<usize> = (0..n_comp).filter(|&k| class[k] == "rejected").collect();
    let cols = |m: &Mat, idx: &[usize]| {
        let mut out = Mat::zeros(m.rows, idx.len());
        for (j, &k) in idx.iter().enumerate() {
            out.col_mut(j).copy_from_slice(m.col(k));
        }
        out
    };

    // reporting.quality_metrics: rejected components' fit to accepted ones, variance summary.
    let varex = s.table.col("variance explained").unwrap().to_vec();
    let mut rej_on_acc = vec![f64::NAN; n_comp];
    let total_rej = if acc.is_empty() || rej.is_empty() {
        f64::NAN
    } else {
        let (ra, aa) = (cols(&mixing, &rej), cols(&mixing, &acc));
        let r = residual(&ra, &aa);
        for (j, &k) in acc.iter().enumerate() {
            let sse = r.col(j).iter().fold(-0.0, |a, v| a + v * v);
            let mean = sum(aa.col(j)) / aa.rows as f64;
            let ss = (0..aa.rows).fold(-0.0, |a, t| a + (aa[(t, j)] - mean) * (aa[(t, j)] - mean));
            rej_on_acc[k] = 100.0 * (1.0 - sse / ss);
        }
        sum(&acc
            .iter()
            .map(|&k| rej_on_acc[k] * varex[k])
            .collect::<Vec<_>>())
            / 100.0
    };
    let label_sum = |label: &str| {
        sum(&(0..n_comp)
            .filter(|&k| class[k] == label)
            .map(|k| varex[k])
            .collect::<Vec<_>>())
    };
    let betas = row_coeffs(&clf_optcom, n_vol, &mixing, |y| demean(y));
    let (mut sst, mut sse, mut kept) = (0.0, 0.0, 0.0);
    let mut dm = vec![0.0; n_vol];
    for r in 0..clf_pos.len() {
        dm.copy_from_slice(&clf_optcom[r * n_vol..(r + 1) * n_vol]);
        demean(&mut dm);
        for t in 0..n_vol {
            let all: f64 = (0..n_comp).map(|k| betas[(r, k)] * mixing[(t, k)]).sum();
            let low: f64 = rej.iter().map(|&k| betas[(r, k)] * mixing[(t, k)]).sum();
            sst += dm[t] * dm[t];
            sse += (dm[t] - all) * (dm[t] - all);
            kept += (dm[t] - low) * (dm[t] - low);
        }
    }
    let total_r2 = (1.0 - sse / sst) * 100.0;
    for (k, v) in [
        (
            "total_var_exp_rejected_components_on_accepted",
            Json::Num(total_rej),
        ),
        ("accepted_variance", Json::Num(label_sum("accepted"))),
        ("rejected_variance", Json::Num(label_sum("rejected"))),
        ("unmodeled_variance", Json::Num(100.0 - total_r2)),
        ("retained_variance", Json::Num(kept / sst * 100.0)),
    ] {
        s.cross.push((k.into(), v));
    }

    log("Writing tables");
    // ICA tables and maps.
    let has = |m: &str| tree_needs(&s.tree, m);
    let mut ica_cols = vec![component_column("ICA", n_comp)];
    ica_cols.extend(metric_tsv(
        &metrics,
        &metrics.normalized_variance_explained,
        has("countnoise") || has("d_table_score"),
        has("d_table_score"),
    ));
    let base: Vec<&str> = metric_columns(&metrics).iter().map(|(n, _)| *n).collect();
    // generate_metrics' preferred order puts HFC before optimal sign; other new metrics follow.
    if let Some(hfc) = s.table.col("HFC") {
        let at = ica_cols
            .iter()
            .position(|(n, _)| n == "optimal sign")
            .unwrap();
        ica_cols.insert(at, ("HFC".into(), Column::Float(hfc.to_vec())));
    }
    for (n, v) in s
        .table
        .columns
        .iter()
        .filter(|(n, _)| !base.contains(&n.as_str()) && n != "HFC")
    {
        ica_cols.push((n.clone(), Column::Float(v.clone())));
    }
    ica_cols.push(("classification".into(), Column::Text(class.clone())));
    ica_cols.push((
        "classification_tags".into(),
        Column::Text(s.table.tags.iter().map(|t| t.join(",")).collect()),
    ));
    // quality_metrics inserts it before the classification columns, or appends it when no
    // component is accepted or none rejected.
    let rej_col = (
        "Var Exp of rejected to accepted".into(),
        Column::Float(rej_on_acc),
    );
    let at = ica_cols.len()
        - if acc.is_empty() || rej.is_empty() {
            0
        } else {
            2
        };
    ica_cols.insert(at, rej_col);
    // The reports step appends these to the written metrics table and its description.
    let correlations = match (&external, opts.no_reports) {
        (Some(reg), false) => external::correlations(reg, &mixing),
        _ => Vec::new(),
    };
    ica_cols.extend(
        correlations
            .iter()
            .map(|(n, v)| (n.clone(), Column::Float(v.clone()))),
    );
    w.tsv(
        "ICA mixing tsv",
        &name("desc-ICA_mixing.tsv"),
        &mixing_columns("ICA", &mixing, n_comp),
    )?;
    // tedana describes the table as generate_metrics returned it, plus the tags column that
    // automatic_selection adds to that same DataFrame.
    let mut ica_names: Vec<&str> = ica_cols
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| base.contains(n) || *n == "HFC")
        .collect();
    ica_names.push("classification_tags");
    let mut ica_meta = metadata(&ica_names);
    if let Json::Obj(kv) = &mut ica_meta {
        for (col, _) in &correlations {
            let reg = col.trim_start_matches("external regressor correlation ");
            kv.push((
                col.clone(),
                obj(vec![
                    ("LongName", format!("External regressor correlation ({reg})").as_str().into()),
                    ("Description", format!("Pearson correlation coefficient between the component time series and the external regressor '{reg}'.").as_str().into()),
                    ("Units", "Pearson correlation coefficient".into()),
                ]),
            ));
        }
    }
    w.json(
        "ICA metrics json",
        &name("desc-tedana_metrics.json"),
        &ica_meta,
    )?;
    w.tsv(
        "ICA metrics tsv",
        &name("desc-tedana_metrics.tsv"),
        &ica_cols,
    )?;
    let mut status = vec![component_column("ICA", n_comp)];
    status.extend(
        s.status
            .iter()
            .map(|(n, v)| (n.clone(), Column::Text(v.clone()))),
    );
    w.tsv(
        "ICA status table tsv",
        &name("desc-ICA_status_table.tsv"),
        &status,
    )?;
    w.json(
        "ICA cross component metrics json",
        &name("desc-ICACrossComponent_metrics.json"),
        &Json::Obj(s.cross.clone()),
    )?;
    w.json(
        "ICA decision tree json",
        &name("desc-ICA_decision_tree.json"),
        &s.tree,
    )?;
    let decomp = decomposition(
        "Independent components analysis with FastICA algorithm implemented by sklearn. ",
        (0..n_comp).map(|k| component_name("ICA", k, n_comp)),
        "ICA fit to dimensionally-reduced optimally combined data.",
    );
    w.json(
        "ICA decomposition json",
        &name("desc-ICA_decomposition.json"),
        &decomp,
    )?;

    // io.writeresults: component maps on all base-mask voxels, z-scored maps and denoising on the
    // denoising mask.
    log("Writing component maps and the denoised series");
    let (denoise_pos, denoise_optcom) = subset(1);
    let dc = &[dims[0], dims[1], dims[2], n_comp];
    let pes = row_coeffs(&c.optcom, n_vol, &mixing, |_| {});
    w.image(
        "ICA components img",
        &name("desc-ICA_components.nii.gz"),
        dc,
        Voxels::F32(&grid(n_grid, epi, n_comp, |v, k| pes[(v, k)])),
    )?;
    let z = row_coeffs(&denoise_optcom, n_vol, &zscore_columns(&mixing), |y| {
        zscore(y)
    });
    w.image(
        "z-scored ICA components img",
        &name("desc-ICA_stat-z_components.nii.gz"),
        dc,
        Voxels::F32(&grid(n_grid, &denoise_pos, n_comp, |v, k| z[(v, k)])),
    )?;
    if !acc.is_empty() {
        let da = &[dims[0], dims[1], dims[2], acc.len()];
        w.image(
            "ICA accepted components img",
            &name("desc-ICAAccepted_components.nii.gz"),
            da,
            Voxels::F32(&grid(n_grid, epi, acc.len(), |v, j| pes[(v, acc[j])])),
        )?;
        w.image(
            "z-scored ICA accepted components img",
            &name("desc-ICAAccepted_stat-z_components.nii.gz"),
            da,
            Voxels::F32(&grid(n_grid, &denoise_pos, acc.len(), |v, j| {
                z[(v, acc[j])]
            })),
        )?;
    }
    let low = row_coeffs(&denoise_optcom, n_vol, &mixing, |y| demean(y));
    let denoised = grid(n_grid, &denoise_pos, n_vol, |v, t| {
        denoise_optcom[v * n_vol + t]
            - rej
                .iter()
                .map(|&k| low[(v, k)] * mixing[(t, k)])
                .sum::<f64>()
    });
    w.image(
        "denoised ts img",
        &name("desc-denoised_bold.nii.gz"),
        d4,
        Voxels::F32(&denoised),
    )?;

    let description = obj(vec![
        ("Name", "meica Outputs".into()),
        ("BIDSVersion", "1.5.0".into()),
        ("DatasetType", "derivative".into()),
        (
            "GeneratedBy",
            Json::Arr(vec![obj(vec![
                ("Name", "meica".into()),
                ("Version", VERSION.into()),
                (
                    "Description",
                    "A Rust port of tedana's multi-echo ICA denoising (tedana e07066c).".into(),
                ),
                ("CodeURL", "https://github.com/niivue/meica-rs".into()),
                ("Command", opts.command.as_str().into()),
            ])]),
        ),
    ]);
    w.json(
        "data description json",
        &name("dataset_description.json"),
        &description,
    )?;
    let files = w.finish(&name("desc-tedana_registry.json"), input_names)?;
    let summary = Summary {
        n_components: n_comp,
        accepted: acc.len(),
        rejected: rej.len(),
        seed: ica_seed,
        n_iter,
    };
    Ok((summary, files))
}

/// tedana's non-aggressive denoising applied to another series of the same run (e.g. one resampled
/// to an anatomical space): per voxel, fit every column of `desc-ICA_mixing.tsv` to the demeaned
/// series after `dummies` volumes and subtract the part of the columns `desc-tedana_metrics.tsv`
/// classifies as rejected. The dummy volumes pass through. Float32, `bold`'s header otherwise.
pub fn denoise(
    bold: &Nifti,
    mixing_tsv: &str,
    metrics_tsv: &str,
    dummies: usize,
) -> Result<Vec<u8>, String> {
    // CRLF-tolerant: a metrics row ending "rejected\r" must still count
    let rows_of = |text: &str| -> Vec<Vec<String>> {
        text.lines()
            .map(|l| l.trim_end_matches('\r'))
            .filter(|l| !l.is_empty())
            .map(|l| l.split('\t').map(String::from).collect())
            .collect()
    };
    let mix = rows_of(mixing_tsv);
    let names = mix.first().ok_or("empty mixing TSV")?;
    let mut values = Vec::new();
    for (r, row) in mix.iter().enumerate().skip(1) {
        if row.len() != names.len() {
            return Err(format!(
                "mixing TSV row {r} has {} cells, the header {}",
                row.len(),
                names.len()
            ));
        }
        for v in row {
            values.push(
                v.parse::<f64>()
                    .map_err(|_| format!("mixing TSV: {v} is not a number"))?,
            );
        }
    }
    let mixing = Mat::from_row_major(mix.len() - 1, names.len(), &values);
    let metrics = rows_of(metrics_tsv);
    let head = metrics.first().ok_or("empty metrics TSV")?;
    let col = |c: &str| {
        head.iter()
            .position(|h| h == c)
            .ok_or(format!("metrics TSV has no {c} column"))
    };
    let (comp, class) = (col("Component")?, col("classification")?);
    let mut rejected = Vec::new();
    for cells in &metrics[1..] {
        if cells.get(class).map(String::as_str) == Some("rejected") {
            let name = cells
                .get(comp)
                .ok_or("metrics TSV row without a Component")?;
            let k = names
                .iter()
                .position(|n| n == name)
                .ok_or(format!("{name} is not in the mixing TSV"))?;
            if !rejected.contains(&k) {
                rejected.push(k);
            }
        }
    }
    let [nx, ny, nz, nt] = bold.dims;
    if nt != dummies + mixing.rows {
        return Err(format!(
            "{nt} volumes minus {dummies} dummy scans do not match the {} mixing rows",
            mixing.rows
        ));
    }
    if !bold.header.little_endian {
        return Err("big-endian NIfTI is not supported".into());
    }
    let n_vox = nx * ny * nz;
    let betas = coeffs(
        |v, y| {
            y.clear();
            y.extend((dummies..nt).map(|t| bold.value(v + n_vox * t)));
            demean(y);
        },
        n_vox,
        &mixing,
        false,
    );
    let mut out = bold.header_bytes().to_vec();
    out[70..72].copy_from_slice(&16i16.to_le_bytes()); // float32
    out[72..74].copy_from_slice(&32i16.to_le_bytes());
    out[108..112].copy_from_slice(&352f32.to_le_bytes()); // vox_offset: extensions dropped
    out[112..116].copy_from_slice(&1f32.to_le_bytes()); // scl_slope
    out[116..120].copy_from_slice(&0f32.to_le_bytes()); // scl_inter
    out.extend([0u8; 4]);
    // ponytail: whole image in memory (~4 bytes per value); stream volumes if browser memory binds.
    out.reserve(n_vox * nt * 4);
    for t in 0..nt {
        for v in 0..n_vox {
            let mut x = bold.value(v + n_vox * t);
            if t >= dummies {
                x -= rejected
                    .iter()
                    .map(|&k| betas[(v, k)] * mixing[(t - dummies, k)])
                    .sum::<f64>();
            }
            out.extend((x as f32).to_le_bytes());
        }
    }
    Ok(out)
}
