//! `meica`: tedana's command line for the ported default workflow. Flag names and defaults
//! follow tedana; a flag whose non-default behaviour is not ported is an error, never ignored.

use meica_core::io::{Nifti, header_of, parse_nifti, read_nifti};
use meica_core::mapca::Criterion;
use meica_core::moco;
use meica_core::output::{Writer, gzip};
use meica_core::selection::{bundled_tree, load_tree, takes_regressors};
use meica_core::workflow::{Options, VERSION, denoise, output_name, run};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "usage: meica -d FILE [FILE ...] -e TE [TE ...] [--out-dir DIR] [--prefix P]
              [--tree tedana_orig|meica|minimal|demo_external_*|motion_aroma|motion_f|FILE]
              [--external TSV] [--mask FILE] [--dummy-scans N] [--moco] [--tedpca aic|kic|mdl]
              [--seed N] [--maxit N] [--maxrestart N] [--n-threads N] [--no-reports] [--quiet]
       meica --denoise BOLD --mixing TSV --metrics TSV -o OUT [--dummy-scans N]

Multi-echo ICA denoising: tedana's default workflow (e07066c) in Rust.
TEs are seconds (< 1) or milliseconds. Outputs use tedana's BIDS names.
Accepted at their tedana defaults only: --convention bids, --masktype dropout,
--fittype loglin, --combmode t2s, --ica-method fastica, --backend cpu, --gscontrol none,
--n-robust-runs 30. --external takes regressors for trees with an
external_regressor_config; --no-reports only drops their correlation columns (no figures are
made). Accepted and ignored: --png-cmap, --overwrite/-f, --debug.
--moco (not a tedana flag) first motion-corrects the echoes with niimath (found next to meica,
then on PATH): echo 1's BOLD reference, dummy count and motion, applied to every echo in one
interpolation. brainchop-mindgrab masks the reference unless --mask is given, and niimath writes
fMRIPrep-named confounds (desc-mocoConfounds_timeseries.tsv), used as --external by trees that
take regressors. A non-zero --dummy-scans overrides niimath's count.
--denoise (not a tedana flag) applies a finished run's classification to another 4D series of
the same run, such as one resampled to an anatomical space: tedana's non-aggressive denoising
with desc-ICA_mixing.tsv and desc-tedana_metrics.tsv. BOLD keeps its first N volumes (the run's
--dummy-scans) unchanged; the rest must match the mixing rows. OUT is float32, gzipped if it
ends in .gz.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--denoise") {
        return match denoise_cli(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("meica: {e}");
                ExitCode::FAILURE
            }
        };
    }
    match parse(&args).and_then(|(inputs, out_dir, mut opts, quiet, moco)| {
        let start = std::time::Instant::now();
        let mut log = |m: &str| {
            if !quiet {
                eprintln!("[{:6.1}s] {m}", start.elapsed().as_secs_f64());
            }
        };
        let (echoes, confounds) = if moco {
            let (e, c) = moco_echoes(&inputs, &mut opts, &mut log)?;
            (e, Some(c))
        } else {
            log("Loading input data");
            let e = inputs
                .iter()
                .map(|p| read_nifti(p))
                .collect::<Result<_, _>>()?;
            (e, None)
        };
        let writer = Writer::to_dir(&out_dir)?;
        let names: Vec<String> = inputs.iter().map(|p| relative(&out_dir, p)).collect();
        let (s, _) = run(echoes, &names, writer, &opts, &mut log)?;
        if let Some(c) = confounds {
            let saved = out_dir.join(output_name(
                &opts.prefix,
                "desc-mocoConfounds_timeseries.tsv",
            ));
            std::fs::write(&saved, c).map_err(|e| format!("{}: {e}", saved.display()))?;
        }
        log(&format!(
            "{} components: {} accepted, {} rejected (seed {}, {} ICA iterations). Outputs in {}",
            s.n_components,
            s.accepted,
            s.rejected,
            s.seed,
            s.n_iter,
            out_dir.display()
        ));
        Ok(())
    }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) if e == "version" => {
            println!("meica {VERSION}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("meica: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `--moco`: run niimath (and brainchop-mindgrab) in a temporary folder and read the corrected
/// echoes from niimath's stdout; fills in the mask, dummy count and regressors left unset.
/// Returns the echoes and niimath's confounds, saved only once the workflow succeeds.
fn moco_echoes(
    inputs: &[PathBuf],
    opts: &mut Options,
    log: &mut dyn FnMut(&str),
) -> Result<(Vec<Nifti>, Vec<u8>), String> {
    if inputs.len() != opts.tes.len() {
        return Err(format!(
            "{} echoes but {} TEs",
            inputs.len(),
            opts.tes.len()
        ));
    }
    let dir = private_temp_dir()?;
    let result = (|| {
        let d = dir.to_str().ok_or("temporary folder path is not UTF-8")?;
        let path = |p: &PathBuf| {
            p.to_str()
                .map(String::from)
                .ok_or("input path is not UTF-8")
        };
        let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"));
        log("Motion estimate (niimath)");
        tool(&moco::estimate(&path(&inputs[0])?, d))?;
        if opts.dummy_scans == 0 {
            opts.dummy_scans = moco::dummy_scans(&String::from_utf8_lossy(&read(moco::NSS)?))?;
        }
        match &opts.mask {
            Some(bytes) => {
                std::fs::write(dir.join(moco::MASK), bytes).map_err(|e| e.to_string())?
            }
            None => {
                log("Brain mask (brainchop-mindgrab)");
                tool(&moco::mask(d))?;
                opts.mask = Some(read(moco::MASK)?);
            }
        }
        let tr = header_of(std::fs::File::open(&inputs[0]).map_err(|e| e.to_string())?)
            .map_err(|e| format!("{}: {e}", inputs[0].display()))?
            .pixdim[4];
        let echoes = (inputs.iter().enumerate())
            .map(|(i, p)| {
                log(&format!("Motion correction of echo {} (niimath)", i + 1));
                let confounds = (i == 0).then_some((tr, opts.dummy_scans));
                parse_nifti(tool(&moco::apply(&path(p)?, d, "-", confounds))?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let confounds = read(moco::CONFOUNDS)?;
        if opts.external.is_none() && takes_regressors(&load_tree(&opts.tree)?) {
            opts.external = Some(String::from_utf8_lossy(&confounds).into_owned());
        }
        Ok((echoes, confounds))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// A new temporary folder, created exclusively (never an existing one, which we would delete)
/// and private to this user: it holds the subject's scans.
fn private_temp_dir() -> Result<PathBuf, String> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let dir = std::env::temp_dir().join(format!("meica-moco-{}-{nanos:x}", std::process::id()));
    builder
        .create(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

/// Run a tool found next to this executable, else on PATH; its stdout on success.
fn tool(argv: &[String]) -> Result<Vec<u8>, String> {
    let name = &argv[0];
    let beside = std::env::current_exe()
        .ok()
        .map(|e| e.with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX)))
        .filter(|p| p.is_file());
    let out = Command::new(beside.unwrap_or_else(|| name.into()))
        .args(&argv[1..])
        .env("FSLOUTPUTTYPE", "NIFTI_GZ") // as the browser driver: see core/src/moco.rs
        .output()
        .map_err(|e| format!("{name}: {e} (--moco needs it next to meica or on PATH)"))?;
    if !out.status.success() {
        let text = String::from_utf8_lossy(if out.stderr.is_empty() {
            &out.stdout
        } else {
            &out.stderr
        });
        return Err(format!("{name} failed: {}", text.trim()));
    }
    Ok(out.stdout)
}

fn parse(args: &[String]) -> Result<(Vec<PathBuf>, PathBuf, Options, bool, bool), String> {
    let mut opts = Options {
        command: std::iter::once("meica".to_string())
            .chain(args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" "),
        ..Options::default()
    };
    let (mut inputs, mut out_dir, mut quiet, mut moco) =
        (Vec::new(), PathBuf::from("."), false, false);
    let mut i = 0;
    // Values up to the next flag.
    let values = |i: &mut usize| -> Vec<String> {
        let start = *i + 1;
        let mut end = start;
        while end < args.len() && !(args[end].starts_with('-') && args[end].parse::<f64>().is_err())
        {
            end += 1;
        }
        *i = end;
        args[start..end].to_vec()
    };
    let one = |i: &mut usize, flag: &str| -> Result<String, String> {
        let v = values(i);
        if v.len() == 1 {
            Ok(v[0].clone())
        } else {
            Err(format!("{flag} takes one value"))
        }
    };
    let number = |s: String, flag: &str| {
        s.parse::<usize>()
            .map_err(|_| format!("{flag} must be a non-negative integer, not {s}"))
    };
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-h" | "--help" => return Err("help".into()),
            "-v" | "--version" => return Err("version".into()),
            "-d" => inputs = values(&mut i).into_iter().map(PathBuf::from).collect(),
            "-e" => {
                opts.tes = values(&mut i)
                    .iter()
                    .map(|s| {
                        s.parse::<f64>()
                            .map_err(|_| format!("-e: {s} is not a number"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--out-dir" => out_dir = PathBuf::from(one(&mut i, flag)?),
            "--prefix" => opts.prefix = one(&mut i, flag)?,
            "--tree" => {
                let t = one(&mut i, flag)?;
                // Bundled names first, as tedana (a file named "meica" is the binary itself).
                opts.tree = if bundled_tree(&t).is_none() && Path::new(&t).is_file() {
                    std::fs::read_to_string(&t).map_err(|e| format!("{t}: {e}"))?
                } else {
                    t
                };
            }
            "--tedpca" => opts.criterion = Criterion::parse(&one(&mut i, flag)?)?,
            "--seed" => {
                let s = one(&mut i, flag)?;
                opts.seed = s.parse().map_err(|_| format!("--seed must be a non-negative integer (tedana's -1 is not supported), not {s}"))?;
            }
            "--maxit" => opts.maxit = number(one(&mut i, flag)?, flag)?,
            "--maxrestart" => opts.maxrestart = number(one(&mut i, flag)?, flag)?,
            "--n-threads" => {
                let n = number(one(&mut i, flag)?, flag)?;
                if n > 0 {
                    meica_core::set_threads(n)?;
                }
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--external" => {
                let f = one(&mut i, flag)?;
                opts.external = Some(std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?);
            }
            "--mask" => {
                let f = one(&mut i, flag)?;
                opts.mask = Some(std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?);
            }
            "--dummy-scans" => opts.dummy_scans = number(one(&mut i, flag)?, flag)?,
            "--moco" => {
                moco = true;
                i += 1;
            }
            "--no-reports" => {
                opts.no_reports = true;
                i += 1;
            }
            "--overwrite" | "-f" | "--debug" => i += 1,
            "--png-cmap" => {
                one(&mut i, flag)?;
            }
            "--convention" | "--masktype" | "--fittype" | "--combmode" | "--ica-method"
            | "--backend" | "--gscontrol" | "--n-robust-runs" => {
                let default = match flag {
                    "--convention" => "bids",
                    "--masktype" => "dropout",
                    "--fittype" => "loglin",
                    "--combmode" => "t2s",
                    "--ica-method" => "fastica",
                    "--backend" => "cpu",
                    "--gscontrol" => "none",
                    _ => "30",
                };
                let v = values(&mut i);
                if v != [default] {
                    return Err(format!(
                        "{flag} {} is not ported; only {default}",
                        v.join(" ")
                    ));
                }
            }
            other => return Err(format!("{other} is not supported by meica\n\n{USAGE}")),
        }
    }
    if inputs.is_empty() || opts.tes.is_empty() {
        return Err(format!("-d and -e are required\n\n{USAGE}"));
    }
    Ok((inputs, out_dir, opts, quiet, moco))
}

/// `meica --denoise BOLD --mixing TSV --metrics TSV -o OUT [--dummy-scans N]`.
fn denoise_cli(args: &[String]) -> Result<(), String> {
    let get = |flag: &str| {
        let i = args.iter().position(|a| a == flag)?;
        args.get(i + 1)
    };
    let known = ["--denoise", "--mixing", "--metrics", "-o", "--dummy-scans"];
    if let Some(a) = args
        .iter()
        .step_by(2)
        .find(|a| !known.contains(&a.as_str()))
    {
        return Err(format!("{a} is not a --denoise option\n\n{USAGE}"));
    }
    let need = |flag: &str| get(flag).ok_or(format!("--denoise needs {flag}\n\n{USAGE}"));
    let text = |f: &String| std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"));
    let dummies = get("--dummy-scans").map_or(Ok(0), |s| {
        s.parse()
            .map_err(|_| format!("--dummy-scans must be a non-negative integer, not {s}"))
    })?;
    let out = need("-o")?;
    let bold = read_nifti(Path::new(need("--denoise")?))?;
    let bytes = denoise(
        &bold,
        &text(need("--mixing")?)?,
        &text(need("--metrics")?)?,
        dummies,
    )?;
    let bytes = if out.ends_with(".gz") {
        gzip(&bytes)
    } else {
        bytes
    };
    std::fs::write(out, bytes).map_err(|e| format!("{out}: {e}"))
}

/// `os.path.relpath(target, start=base)` on canonical paths, as tedana's registry lists inputs.
fn relative(base: &Path, target: &Path) -> String {
    let (b, t) = (
        base.canonicalize().unwrap_or(base.into()),
        target.canonicalize().unwrap_or(target.into()),
    );
    let (bc, tc): (Vec<_>, Vec<_>) = (b.components().collect(), t.components().collect());
    let common = bc.iter().zip(&tc).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<String> = vec!["..".into(); bc.len() - common];
    parts.extend(
        tc[common..]
            .iter()
            .map(|c| c.as_os_str().to_string_lossy().into_owned()),
    );
    parts.join("/")
}
