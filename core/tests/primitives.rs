//! Core primitives against numpy/scipy/mpmath vectors from `validation/gen_fixtures.py`.

use meica_core::{
    fft,
    json::Json,
    linalg,
    linalg::Mat,
    mask, math,
    rng::{Pcg64, RandomState},
    stats,
};

fn fixture(name: &str) -> Json {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    Json::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn v(j: &Json, k: &str) -> Vec<f64> {
    j.get(k)
        .and_then(Json::f64s)
        .unwrap_or_else(|| panic!("missing {k}"))
}

fn mat(j: &Json) -> Mat {
    let n = |k| j.get(k).unwrap().as_f64().unwrap() as usize;
    Mat::from_row_major(n("rows"), n("cols"), &v(j, "data"))
}

/// Distance in units in the last place (same-sign finite values).
fn ulps(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let key = |x: f64| {
        let i = x.to_bits() as i64;
        if i < 0 { i64::MIN - i } else { i }
    };
    key(a).abs_diff(key(b))
}

fn worst_ulps(f: fn(f64) -> f64, case: &Json) -> (u64, u64) {
    let (x, mp, np) = (v(case, "x"), v(case, "mp"), v(case, "np"));
    let mut worst = (0, 0);
    for i in 0..x.len() {
        let y = f(x[i]);
        worst = (worst.0.max(ulps(y, mp[i])), worst.1.max(ulps(y, np[i])));
    }
    worst
}

/// Bounds: exp is msun's < 1 ulp; ln is correctly rounded (gauss draws need numpy's bits);
/// tanh is msun's < 2 ulp, and numpy's own tanh is not correctly rounded on 16% of these points.
#[test]
fn elementary_functions_within_bounds() {
    let m = fixture("math");
    for (name, f, bound) in [
        ("exp", math::exp as fn(f64) -> f64, 1),
        ("ln", math::ln, 0),
        ("tanh", math::tanh, 2),
    ] {
        let (vs_mp, vs_np) = worst_ulps(f, m.get(name).unwrap());
        println!("{name}: worst {vs_mp} ulp vs mpmath, {vs_np} ulp vs numpy");
        assert!(vs_mp <= bound && vs_np <= bound.max(1), "{name}");
    }
}

#[test]
fn lgamma_and_sincos() {
    let m = fixture("math");
    let g = m.get("lgamma").unwrap();
    let (x, mp) = (v(g, "x"), v(g, "mp"));
    let worst = (0..x.len())
        .map(|i| (math::lgamma(x[i]) - mp[i]).abs() / mp[i].abs().max(1.0))
        .fold(0.0, f64::max);
    // Absolute error near the roots at 1 and 2 comes from the recurrence up to x >= 8.
    println!("lgamma: worst error {worst:e} (relative, absolute below 1)");
    assert!(worst < 1e-14);
    let sc = m.get("sincos").unwrap();
    let (k, n, s, c) = (v(sc, "k"), v(sc, "n"), v(sc, "sin"), v(sc, "cos"));
    let mut worst = 0.0f64;
    for i in 0..k.len() {
        let (ys, yc) = math::sincos_2pi(k[i] as u64, n[i] as u64);
        worst = worst.max((ys - s[i]).abs()).max((yc - c[i]).abs());
    }
    println!("sincos_2pi: worst abs error {worst:e}");
    assert!(worst <= 2.0 * f64::EPSILON);
}

fn worst_rel(got: impl Iterator<Item = f64>, want: &[f64], floor: f64) -> f64 {
    got.zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(floor))
        .fold(0.0, f64::max)
}

#[test]
fn distributions_match_scipy() {
    let s = fixture("stats");
    let b = s.get("betainc").unwrap();
    let (a, bb, x) = (v(b, "a"), v(b, "b"), v(b, "x"));
    let e = worst_rel(
        (0..a.len()).map(|i| stats::betainc(a[i], bb[i], x[i])),
        &v(b, "y"),
        1e-300,
    );
    println!("betainc: worst relative {e:e}");
    assert!(e < 5e-13);
    let t = s.get("stdtr").unwrap();
    let (df, tv, p) = (v(t, "df"), v(t, "t"), v(t, "p"));
    // df <= 1000 covers t_to_z (df = volumes - 2). Larger df only reaches the reported
    // signal-noise_p column, where the continued fraction's slow convergence costs ~1e-12.
    for (max_df, bound) in [(1000.0, 1e-13), (f64::INFINITY, 2e-12)] {
        let idx: Vec<usize> = (0..df.len()).filter(|&i| df[i] <= max_df).collect();
        let want: Vec<f64> = idx.iter().map(|&i| p[i]).collect();
        let e = worst_rel(
            idx.iter().map(|&i| stats::t_cdf(tv[i], df[i])),
            &want,
            1e-300,
        );
        println!("t_cdf df <= {max_df}: worst relative {e:e}");
        assert!(e < bound);
    }
    let f = s.get("f_ppf").unwrap();
    let (d, q) = (v(f, "dfd"), v(f, "q"));
    let e = worst_rel(
        (0..d.len()).map(|i| stats::f_ppf_dfn1(q[i], d[i])),
        &v(f, "f"),
        1.0,
    );
    println!("f_ppf: worst relative {e:e}");
    assert!(e < 1e-13);
}

/// Columns equal up to sign.
fn assert_cols_match(got: &Mat, want: &Mat, cols: usize, tol: f64, what: &str) {
    for j in 0..cols {
        let sign = if linalg::dot(got.col(j), want.col(j)) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let e = got
            .col(j)
            .iter()
            .zip(want.col(j))
            .map(|(g, w)| (sign * g - w).abs())
            .fold(0.0, f64::max);
        assert!(e < tol, "{what} column {j}: {e:e}");
    }
}

#[test]
fn svd_eigh_lstsq_match_lapack() {
    let l = fixture("linalg");
    for case in l.get("svd").unwrap().as_arr().unwrap() {
        let a = mat(case.get("a").unwrap());
        let (u, s, vt) = linalg::svd(&a);
        let want = v(case, "s");
        let e = worst_rel(s.iter().copied(), &want, want[0]);
        println!(
            "svd {}x{}: worst singular value error {e:e} of s_max",
            a.rows, a.cols
        );
        assert!(e < 1e-13);
        // Vector error scales with 1 / (relative gap) to the nearest singular value.
        let tol = 1e-13 * want[0]
            / (0..want.len() - 1)
                .map(|i| want[i] - want[i + 1])
                .fold(f64::INFINITY, f64::min);
        assert_cols_match(&u, &mat(case.get("u").unwrap()), s.len(), tol, "u");
        assert_cols_match(
            &vt.t(),
            &mat(case.get("vt").unwrap()).t(),
            s.len(),
            tol,
            "v",
        );
    }
    for case in l.get("eigh").unwrap().as_arr().unwrap() {
        let (w, vecs) = linalg::eigh(&mat(case.get("a").unwrap()));
        let want = v(case, "w");
        let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let e = worst_rel(w.iter().copied(), &want, scale);
        println!("eigh {}: worst eigenvalue error {e:e} of max |w|", w.len());
        assert!(e < 1e-14);
        assert_cols_match(
            &vecs,
            &mat(case.get("v").unwrap()),
            w.len(),
            1e-12,
            "eigh v",
        );
    }
    let ls = l.get("lstsq").unwrap();
    let b = linalg::lstsq(&mat(ls.get("x").unwrap()), &mat(ls.get("y").unwrap()));
    let want = mat(ls.get("b").unwrap());
    let e = worst_rel(b.data.iter().copied(), &want.data, 1.0);
    println!("lstsq: worst error {e:e}");
    assert!(e < 1e-14);
}

#[test]
fn fft_matches_pocketfft() {
    let f = fixture("fft");
    let check = |re: &[f64], im: &[f64], fre: &[f64], fim: &[f64], what: String| {
        let peak = fre.iter().chain(fim).fold(0.0f64, |m, x| m.max(x.abs()));
        let e = re
            .iter()
            .zip(fre)
            .chain(im.iter().zip(fim))
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f64::max)
            / peak;
        assert!(e < 1e-14, "{what}: {e:e}");
        e
    };
    let mut worst = 0.0f64;
    for case in f.get("fft").unwrap().as_arr().unwrap() {
        let (mut re, mut im) = (v(case, "re"), v(case, "im"));
        let n = [re.len()];
        fft::fftn(&mut re, &mut im, &n);
        worst = worst.max(check(
            &re,
            &im,
            &v(case, "fre"),
            &v(case, "fim"),
            format!("fft n={}", re.len()),
        ));
        fft::ifftn(&mut re, &mut im, &n);
        check(
            &re,
            &im,
            &v(case, "re"),
            &v(case, "im"),
            format!("ifft n={}", re.len()),
        );
    }
    let c = f.get("fftn").unwrap();
    let shape: Vec<usize> = v(c, "shape").iter().map(|&s| s as usize).collect();
    let mut re = v(c, "x");
    let mut im = vec![0.0; re.len()];
    fft::fftn(&mut re, &mut im, &shape);
    worst = worst.max(check(&re, &im, &v(c, "fre"), &v(c, "fim"), "fftn".into()));
    println!("fft: worst error {worst:e} of peak magnitude");
}

#[test]
fn random_state_matches_numpy_bit_for_bit() {
    let r = fixture("rng");
    let key = v(&r, "key");
    let rs = RandomState::new(42);
    for (i, (&got, &want)) in rs.key.iter().zip(&key[..16]).enumerate() {
        assert_eq!(got as f64, want, "key[{i}]");
    }
    assert_eq!(rs.key[623] as f64, key[16]);
    let mut rs = RandomState::new(42);
    for (i, want) in v(&r, "random_sample").into_iter().enumerate() {
        assert_eq!(
            rs.random_sample().to_bits(),
            want.to_bits(),
            "random_sample[{i}]"
        );
    }
    for name in ["normal_15x15", "normal_13x13"] {
        let mut rs = RandomState::new(42);
        let want = v(&r, name);
        let diff = want
            .iter()
            .filter(|w| rs.normal().to_bits() != w.to_bits())
            .count();
        assert_eq!(diff, 0, "{name}: {diff} of {} draws differ", want.len());
    }
}

#[test]
fn pcg64_matches_numpy_generator() {
    let Json::Obj(seeds) = fixture("pcg64") else {
        panic!()
    };
    let ints = |j: &Json, k: &str| -> Vec<u64> {
        j.get(k)
            .unwrap()
            .as_arr()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as u64)
            .collect()
    };
    for (seed, want) in &seeds {
        let seed: u64 = seed.parse().unwrap();
        let mut g = Pcg64::new(seed);
        // u64 draws exceed f64 precision: compare the exactly representable high bits.
        for w in ints(want, "u64") {
            assert_eq!(g.next_u64() as f64, w as f64, "seed {seed} u64");
        }
        let mut g = Pcg64::new(seed);
        let got: Vec<u64> = g.choice(100, 90).into_iter().map(|v| v as u64).collect();
        assert_eq!(got, ints(want, "choice_100_90"), "seed {seed} choice");
        let mut g = Pcg64::new(seed);
        let got: Vec<u64> = (0..3)
            .flat_map(|_| g.choice(150, 135))
            .map(|v| v as u64)
            .collect();
        assert_eq!(
            got,
            ints(want, "choice_150_135x3"),
            "seed {seed} repeated choice"
        );
    }
}

#[test]
fn json_round_trip() {
    let src = r#"{"a":[1.0,-2.5e-7,NaN,-Infinity],"s":"q\"\\\né😀","t":true,"n":null,"o":{}}"#;
    let j = Json::parse(src).unwrap();
    assert_eq!(j.get("s").unwrap().as_str().unwrap(), "q\"\\\né😀");
    let again = Json::parse(&j.to_string()).unwrap();
    assert_eq!(again.get("s"), j.get("s"));
    assert_eq!(again.to_string(), j.to_string());
    assert!(Json::parse("[1,]").is_err());
    let py = [
        (1e-5, "1e-05"),
        (1.5e-7, "1.5e-07"),
        (1e16, "1e+16"),
        (1.2e300, "1.2e+300"),
        (0.0001, "0.0001"),
        (2.5, "2.5"),
        (5.0, "5.0"),
        (-0.0, "-0.0"),
    ];
    for (v, want) in py {
        assert_eq!(meica_core::json::py_float(v), want);
    }
    let j = Json::parse(r#"{"b": [1, 2.0], "a": {"x": null}}"#).unwrap();
    assert_eq!(
        j.pretty(),
        "{\n    \"a\": {\n        \"x\": null\n    },\n    \"b\": [\n        1,\n        2.0\n    ]\n}"
    );
}

#[test]
fn sums_log32_and_smoothing_match_numpy_bit_for_bit() {
    let im = fixture("image");
    for case in im.get("sum64").unwrap().as_arr().unwrap() {
        assert_eq!(
            stats::sum(&v(case, "x")).to_bits(),
            case.get("s").unwrap().as_f64().unwrap().to_bits()
        );
    }
    for case in im.get("sum32").unwrap().as_arr().unwrap() {
        let x: Vec<f32> = v(case, "x").iter().map(|&a| a as f32).collect();
        assert_eq!(
            stats::sum_f32(&x),
            case.get("s").unwrap().as_f64().unwrap() as f32
        );
    }
    let l = im.get("log32").unwrap();
    let (x, y) = (v(l, "x"), v(l, "y"));
    let wrong = (0..x.len())
        .filter(|&i| math::ln_f32(x[i] as f32) != y[i] as f32)
        .count();
    println!(
        "ln_f32: {wrong} of {} differ from numpy float32 log",
        x.len()
    );
    assert_eq!(wrong, 0);
    // smooth_array input and output are C order over (7, 6, 5); smooth_fwhm1 works in NIfTI order.
    let s = im.get("smooth").unwrap();
    let d = [7, 6, 5];
    let c_to_nifti = |c: usize| (c / 30) + 7 * ((c / 5) % 6 + 6 * (c % 5));
    let (x, y) = (v(s, "x"), v(s, "y"));
    let mut vol = vec![0.0; x.len()];
    for c in 0..x.len() {
        vol[c_to_nifti(c)] = x[c];
    }
    mask::smooth_fwhm1(&mut vol, d);
    let wrong = (0..y.len())
        .filter(|&c| vol[c_to_nifti(c)].to_bits() != y[c].to_bits())
        .count();
    assert_eq!(
        wrong,
        0,
        "smooth_fwhm1: {wrong} of {} differ from nilearn",
        y.len()
    );
}

/// A float32 NIfTI-1 of zeros.
fn nifti(dims: [i16; 4], vox_offset: f32, n_values: usize) -> Vec<u8> {
    let mut h = vec![0u8; 352];
    h[0..4].copy_from_slice(&348i32.to_le_bytes());
    for (a, d) in [4].iter().chain(&dims).enumerate() {
        h[40 + 2 * a..42 + 2 * a].copy_from_slice(&d.to_le_bytes());
    }
    h[70..72].copy_from_slice(&16i16.to_le_bytes());
    h[72..74].copy_from_slice(&32i16.to_le_bytes());
    h[108..112].copy_from_slice(&vox_offset.to_le_bytes());
    h[344..348].copy_from_slice(b"n+1\0");
    h.resize(352 + 4 * n_values, 0);
    h
}

#[test]
fn bad_inputs_are_errors_not_panics() {
    use meica_core::io::parse_nifti;
    use meica_core::output::Writer;
    use meica_core::pipeline::check_te_values;
    use meica_core::workflow::{Options, run};
    assert!(check_te_values(&[]).is_err());
    assert!(check_te_values(&[15.0, f64::INFINITY]).is_err());
    assert!(parse_nifti(nifti([2, 2, 2, 2], 0.0, 16)).is_err());
    assert!(parse_nifti(nifti([i16::MAX; 4], 352.0, 0)).is_err());
    let echo = || parse_nifti(nifti([4, 4, 4, 3], 352.0, 192)).unwrap();
    let go = |echoes, tes: &[f64]| {
        let opts = Options {
            tes: tes.to_vec(),
            ..Options::default()
        };
        run(echoes, &[], Writer::in_memory(), &opts, &mut |_| {})
    };
    assert!(go(vec![echo()], &[15.0]).is_err());
    assert!(go(vec![], &[15.0, 30.0]).is_err());
    // All zeros: empty EPI mask.
    let e = go(vec![echo(), echo(), echo()], &[15.0, 30.0, 45.0])
        .err()
        .unwrap();
    assert!(e.contains("mask is empty"), "{e}");
    // A single voxel.
    let one = || parse_nifti(nifti([1, 1, 1, 3], 352.0, 3)).unwrap();
    assert!(go(vec![one(), one()], &[15.0, 30.0]).is_err());
    // 5D: dim[0] = 5 with dim[5] = 2.
    let mut five = nifti([2, 2, 2, 2], 352.0, 32);
    five[40..42].copy_from_slice(&5i16.to_le_bytes());
    five[50..52].copy_from_slice(&2i16.to_le_bytes());
    assert!(parse_nifti(five).is_err());
    // Hostile JSON: deep nesting, a bad surrogate pair.
    use meica_core::json::Json;
    assert!(Json::parse(&"[".repeat(100_000)).is_err());
    assert!(Json::parse(r#""\ud800A""#).is_err());
}

#[test]
fn big_endian_nifti_reads_as_little_endian() {
    use meica_core::io::parse_nifti;
    // A 2x1x1x2 int16 image with slope 0.5, in each byte order.
    let image = |le: bool| {
        let i16b = |v: i16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let f32b = |v: f32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let mut h = vec![0u8; 352];
        h[0..4].copy_from_slice(&if le {
            348i32.to_le_bytes()
        } else {
            348i32.to_be_bytes()
        });
        for (a, d) in [4i16, 2, 1, 1, 2].iter().enumerate() {
            h[40 + 2 * a..42 + 2 * a].copy_from_slice(&i16b(*d));
        }
        h[70..72].copy_from_slice(&i16b(4));
        h[76 + 16..80 + 16].copy_from_slice(&f32b(2.5));
        h[108..112].copy_from_slice(&f32b(352.0));
        h[112..116].copy_from_slice(&f32b(0.5));
        for v in [-3i16, 7, 300, -32768] {
            h.extend_from_slice(&i16b(v));
        }
        parse_nifti(h).unwrap()
    };
    let (le, be) = (image(true), image(false));
    assert_eq!(be.dims, [2, 1, 1, 2]);
    assert_eq!(be.header.pixdim[4], 2.5);
    let values: Vec<f64> = (0..4).map(|i| be.value(i)).collect();
    assert_eq!(values, [-1.5, 3.5, 150.0, -16384.0]);
    assert_eq!(values, (0..4).map(|i| le.value(i)).collect::<Vec<_>>());
}
