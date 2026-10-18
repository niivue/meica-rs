//! Pipeline stages against tedana's outputs on the published example.
//! Needs MEICA_REFERENCE_DIR with example/, reference/ (tedana's outputs) and stages/
//! (validation/gen_stage_reference.py). MEICA_REQUIRE_REFERENCE=1 makes a missing dir fail.

use meica_core::fastica::{f_ica, pinv};
use meica_core::io::{Nifti, read_nifti};
use meica_core::json::Json;
use meica_core::linalg::Mat;
use meica_core::mapca::Criterion;
use meica_core::metrics::Metrics;
use meica_core::pipeline::{
    Combined, check_te_values, combine, component_metrics, component_table, tedpca,
};
use meica_core::selection::{Selector, load_tree};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// As given on tedana's command line (ms).
const TES: [f64; 3] = [15.2, 35.32, 55.42];

fn reference_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("MEICA_REFERENCE_DIR").map(PathBuf::from);
    if dir.is_none() {
        assert!(
            std::env::var_os("MEICA_REQUIRE_REFERENCE").is_none(),
            "MEICA_REFERENCE_DIR is not set"
        );
        println!("skipped: MEICA_REFERENCE_DIR is not set");
    }
    dir
}

fn load_nifti(dir: &Path, rel: &str) -> Nifti {
    read_nifti(&dir.join(rel)).unwrap()
}

fn read_json(dir: &Path, rel: &str) -> Json {
    Json::parse(&std::fs::read_to_string(dir.join(rel)).unwrap()).unwrap()
}

/// The volume grid, from tedana's EPI mask.
fn grid(dir: &Path) -> [usize; 3] {
    let d = load_nifti(dir, "stages/epi_mask.nii.gz").dims;
    [d[0], d[1], d[2]]
}

/// Metrics of sklearn's mixing (stages/fastica.json).
/// Metrics on tedana's final mixing: its saved, sign-flipped mixing times "optimal sign" undoes the
/// flip. (stages/fastica.json is the seed-42 run, not final when tedana restarts for no BOLD.)
fn ica_metrics(dir: &Path, c: &Combined) -> Metrics {
    let saved = tsv(dir, "reference/desc-ICA_mixing.tsv");
    let sign = &tsv(dir, "reference/desc-tedana_metrics.tsv")["optimal sign"];
    let cols: Vec<Vec<f64>> = (0..sign.len())
        .map(|k| {
            let s: f64 = sign[k].parse().unwrap();
            let col = &saved[&format!("ICA_{k:02}")];
            col.iter().map(|v| v.parse::<f64>().unwrap() * s).collect()
        })
        .collect();
    let mut mixing = Mat::zeros(cols[0].len(), cols.len());
    for (k, col) in cols.iter().enumerate() {
        mixing.col_mut(k).copy_from_slice(col);
    }
    let tes = check_te_values(&TES).unwrap();
    component_metrics(c, grid(dir), &tes, &mixing).0
}

/// A reference image, decoded.
struct Img {
    dims: [usize; 4],
    data: Vec<f64>,
}

fn load(dir: &Path, rel: &str) -> Img {
    let n = load_nifti(dir, rel);
    Img {
        dims: n.dims,
        data: (0..n.dims.iter().product()).map(|i| n.value(i)).collect(),
    }
}

fn stage(dir: &Path) -> &'static Combined {
    static RUN: OnceLock<Combined> = OnceLock::new();
    RUN.get_or_init(|| {
        let echoes: Vec<Nifti> = (1..=3)
            .map(|e| {
                load_nifti(
                    dir,
                    &format!("example/sub-01_task-motor_echo-{e}_bold.nii.gz"),
                )
            })
            .collect();
        combine(&echoes, &check_te_values(&TES).unwrap()).unwrap()
    })
}

/// Per-voxel values must equal tedana's float32 map bit for bit at the EPI-mask voxels.
fn compare(name: &str, got: impl Fn(usize, usize) -> f64, want: &Img, c: &Combined) {
    let nvox = want.dims[0] * want.dims[1] * want.dims[2];
    let (mut exact, mut worst, mut n) = (0usize, 0.0f64, 0usize);
    for (v, &i) in c.voxels.iter().enumerate() {
        for t in 0..want.dims[3] {
            let (g, w) = (got(v, t) as f32 as f64, want.data[i + nvox * t]);
            n += 1;
            exact += (g == w) as usize;
            worst = worst.max((g - w).abs() / w.abs().max(f64::MIN_POSITIVE));
        }
    }
    println!("{name}: {exact} of {n} values bit-identical as f32, worst relative {worst:e}");
    assert_eq!(exact, n, "{name}: worst relative {worst:e}");
}

#[test]
fn masks_decay_and_optcom_match_tedana() {
    let Some(dir) = reference_dir() else { return };
    let c = stage(&dir);
    let epi = load(&dir, "stages/epi_mask.nii.gz");
    let wrong = c
        .epi_mask
        .iter()
        .zip(&epi.data)
        .filter(|(g, w)| **g != (**w != 0.0))
        .count();
    println!(
        "EPI mask: {} voxels, {wrong} differ from nilearn",
        c.voxels.len()
    );
    assert_eq!(wrong, 0);

    let adaptive = load(&dir, "reference/desc-adaptiveGoodSignal_mask.nii.gz");
    let mut want = vec![0i32; c.masksum.len()];
    for (v, &i) in c.voxels.iter().enumerate() {
        want[v] = adaptive.data[i] as i32;
    }
    let wrong = c.masksum.iter().zip(&want).filter(|(g, w)| g != w).count();
    let hist: Vec<usize> = (0..=3)
        .map(|k| c.masksum.iter().filter(|&&m| m == k).count())
        .collect();
    println!("adaptive mask: counts by value {hist:?}, {wrong} differ from tedana");
    assert_eq!(wrong, 0);

    compare(
        "T2*",
        |v, _| c.t2s[v],
        &load(&dir, "reference/T2starmap.nii.gz"),
        c,
    );
    compare(
        "S0",
        |v, _| c.s0[v],
        &load(&dir, "reference/S0map.nii.gz"),
        c,
    );
    let n_vol = c.optcom.len() / c.voxels.len();
    compare(
        "optcom",
        |v, t| c.optcom[v * n_vol + t],
        &load(&dir, "reference/desc-optcom_bold.nii.gz"),
        c,
    );
}

fn worst_rel(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-300))
        .fold(0.0, f64::max)
}

#[test]
fn mapca_and_kept_data_match_tedpca() {
    let Some(dir) = reference_dir() else { return };
    let c = stage(&dir);
    let want = read_json(&dir, "stages/mapca.json");
    let num = |k: &str| want.get(k).unwrap().as_f64().unwrap();
    let crit = |k: &str| want.get(k).unwrap();
    let kept = load(&dir, "stages/kept_data.nii.gz");
    let d = [kept.dims[0], kept.dims[1], kept.dims[2]];
    let t0 = std::time::Instant::now();
    let r = tedpca(c, d, Criterion::Aic).unwrap();
    println!("tedpca: {:?}", t0.elapsed());
    let m = &r.mapca;
    println!(
        "maPCA: samples {}, IID estimates {:?}, depth {}, n {}, AIC/KIC/MDL {}/{}/{}, 90/95% {}/{}",
        r.positions.len(),
        m.iid_subsample_input,
        m.subsample_depth,
        m.effective_samples,
        m.aic.0,
        m.kic.0,
        m.mdl.0,
        m.varex_90,
        m.varex_95
    );
    assert_eq!(r.positions.len() as f64, num("n_samples"));
    assert_eq!(
        m.iid_subsample_input,
        crit("iid_subsample_input").f64s().unwrap()
    );
    assert_eq!(m.subsample_depth as f64, num("subsample_depth"));
    assert_eq!(m.effective_samples, num("effective_samples"));
    for (name, got) in [("aic", &m.aic), ("kic", &m.kic), ("mdl", &m.mdl)] {
        let w = crit(name);
        assert_eq!(
            got.0 as f64,
            w.get("n").unwrap().as_f64().unwrap(),
            "{name}"
        );
        let e = worst_rel(&got.1, &w.get("curve").unwrap().f64s().unwrap());
        println!("{name} curve: worst relative {e:e}");
        assert!(e < 1e-12, "{name} curve");
    }
    assert_eq!(
        (m.varex_90 as f64, m.varex_95 as f64),
        (num("varex_90"), num("varex_95"))
    );
    // LAPACK SVD, BLAS products and pocketfft sit in Python's path, so values match to rounding.
    for (name, got) in [
        ("cumulative_varex", &m.cumulative_varex),
        ("explained_variance", &m.explained_variance),
    ] {
        let e = worst_rel(got, &crit(name).f64s().unwrap());
        println!("{name}: worst relative {e:e}");
        assert!(e < 1e-13, "{name}");
    }
    let comps = crit("components").as_arr().unwrap();
    let mut worst = 0.0f64;
    for (k, row) in comps.iter().enumerate() {
        let row = row.f64s().unwrap();
        for (j, w) in row.iter().enumerate() {
            worst = worst.max((m.components[(k, j)] - w).abs());
        }
    }
    println!("components: worst abs {worst:e}");
    assert!(worst < 1e-12);
    let nvox = d[0] * d[1] * d[2];
    let n_vol = r.kept_data.cols;
    let mut worst = 0.0f64;
    for (row, &i) in r.positions.iter().enumerate() {
        for t in 0..n_vol {
            worst = worst.max((r.kept_data[(row, t)] - kept.data[i + nvox * t]).abs());
        }
    }
    println!("kept_data: worst abs {worst:e}");
    assert!(worst < 1e-11);
}

fn matrix(j: &Json) -> Mat {
    let rows: Vec<Vec<f64>> = j
        .as_arr()
        .unwrap()
        .iter()
        .map(|r| r.f64s().unwrap())
        .collect();
    Mat::from_row_major(rows.len(), rows[0].len(), &rows.concat())
}

fn worst_abs(got: &Mat, want: &Mat) -> f64 {
    assert_eq!((got.rows, got.cols), (want.rows, want.cols));
    got.data
        .iter()
        .zip(&want.data)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max)
}

#[test]
fn fastica_matches_sklearn_and_f_ica() {
    let Some(dir) = reference_dir() else { return };
    let c = stage(&dir);
    let want = read_json(&dir, "stages/fastica.json");
    let r = tedpca(c, grid(&dir), Criterion::Aic).unwrap();
    let t0 = std::time::Instant::now();
    let ica = f_ica(&r.kept_data, r.mapca.aic.0, 42, 500, 10);
    println!(
        "f_ica: {:?}, seed {}, {} iterations, converged {}",
        t0.elapsed(),
        ica.seed,
        ica.fit.n_iter,
        ica.fit.converged
    );
    assert_eq!(ica.seed as f64, want.get("seed").unwrap().as_f64().unwrap());
    assert_eq!(
        ica.fit.n_iter as f64,
        want.get("n_iter").unwrap().as_f64().unwrap()
    );
    for (name, got) in [
        ("whitening", &ica.fit.whitening),
        ("components", &ica.fit.components),
        ("mixing_raw", &pinv(&ica.fit.components)),
        ("mixing", &ica.mixing),
    ] {
        let w = matrix(want.get(name).unwrap());
        let scale = w.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let e = worst_abs(got, &w);
        println!("{name}: worst abs {e:e} (max |value| {scale:e})");
        // BLAS products and LAPACK's SVD sit in sklearn's path; the fixed point amplifies them little.
        assert!(e < 1e-10 * scale, "{name}");
    }
}

/// Columns of a tedana TSV by header name.
fn tsv(dir: &Path, rel: &str) -> std::collections::HashMap<String, Vec<String>> {
    let text = std::fs::read_to_string(dir.join(rel)).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split('\t')
        .map(String::from)
        .collect();
    let mut cols: std::collections::HashMap<String, Vec<String>> =
        header.iter().map(|h| (h.clone(), Vec::new())).collect();
    for line in lines {
        for (h, v) in header.iter().zip(line.split('\t')) {
            cols.get_mut(h).unwrap().push(v.to_string());
        }
    }
    cols
}

#[test]
fn ica_metrics_match_tedana() {
    let Some(dir) = reference_dir() else { return };
    let c = stage(&dir);
    let t0 = std::time::Instant::now();
    let m = ica_metrics(&dir, c);
    println!("metrics: {:?}", t0.elapsed());
    let want = tsv(&dir, "reference/desc-tedana_metrics.tsv");
    let col = |k: &str| {
        want[k]
            .iter()
            .map(|v| v.parse::<f64>().unwrap())
            .collect::<Vec<f64>>()
    };
    let as_f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let signs: Vec<f64> = m.optimal_sign.iter().map(|&s| s as f64).collect();
    for (name, got) in [
        ("optimal sign", signs),
        ("countsigFT2", as_f(&m.countsig_ft2)),
        ("countsigFS0", as_f(&m.countsig_fs0)),
        ("countnoise", as_f(&m.countnoise)),
        ("d_table_score", m.d_table_score.clone()),
    ] {
        println!("{name}: {got:?}");
        assert_eq!(got, col(name), "{name}");
    }
    for (name, got) in [
        ("kappa", &m.kappa),
        ("rho", &m.rho),
        ("variance explained", &m.variance_explained),
        (
            "normalized variance explained",
            &m.normalized_variance_explained,
        ),
        ("dice_FT2", &m.dice_ft2),
        ("dice_FS0", &m.dice_fs0),
        ("signal-noise_t", &m.signal_noise_t),
        ("signal-noise_p", &m.signal_noise_p),
    ] {
        let e = worst_rel(got, &col(name));
        println!("{name}: worst relative {e:e}");
        // p comes from the incomplete beta at large df (see README Traps).
        assert!(
            e < if name == "signal-noise_p" {
                1e-10
            } else {
                1e-12
            },
            "{name}"
        );
    }
}

#[test]
fn tedana_orig_selection_matches_tedana() {
    let Some(dir) = reference_dir() else { return };
    let c = stage(&dir);
    let m = ica_metrics(&dir, c);
    let s = Selector::select(
        load_tree("tedana_orig").unwrap(),
        component_table(&m),
        3,
        100,
    )
    .unwrap();
    let status = tsv(&dir, "reference/desc-ICA_status_table.tsv");
    let names: Vec<&String> = s.status.iter().map(|(n, _)| n).collect();
    println!("status columns: {names:?}");
    assert_eq!(s.status.len() + 1, status.len(), "status table columns");
    for (name, got) in &s.status {
        assert_eq!(got, &status[name], "{name}");
    }
    let want = tsv(&dir, "reference/desc-tedana_metrics.tsv");
    assert_eq!(s.table.classification, want["classification"]);
    for (c, tags) in s.table.tags.iter().enumerate() {
        let mut got = tags.clone();
        let mut w: Vec<String> = want["classification_tags"][c]
            .split(',')
            .map(String::from)
            .collect();
        got.sort();
        w.sort();
        assert_eq!(got, w, "tags of component {c}");
    }
    println!("classification: {:?}", s.table.classification);
    let cross = read_json(&dir, "reference/desc-ICACrossComponent_metrics.json");
    let mut worst = 0.0f64;
    for (k, v) in &s.cross {
        match (v.as_f64(), cross.get(k).and_then(Json::as_f64)) {
            (Some(g), Some(w)) => worst = worst.max((g - w).abs() / w.abs().max(1e-300)),
            _ => assert_eq!(Some(v), cross.get(k), "{k}"),
        }
    }
    println!(
        "cross-component metrics: {} keys, worst relative {worst:e}",
        s.cross.len()
    );
    assert!(worst < 1e-12);
    for name in ["varex kappa ratio", "d_table_score_node20"] {
        let w: Vec<f64> = want[name]
            .iter()
            .map(|v| {
                if v == "n/a" {
                    f64::NAN
                } else {
                    v.parse().unwrap()
                }
            })
            .collect();
        let g = s.table.col(name).unwrap();
        let e = g
            .iter()
            .zip(&w)
            .filter(|(a, b)| !(a.is_nan() && b.is_nan()))
            .map(|(a, b)| (a - b).abs() / b.abs())
            .fold(0.0, f64::max);
        println!("{name}: worst relative {e:e}");
        assert!(e < 1e-12, "{name}");
    }
}
