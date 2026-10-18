//! `meica`: tedana's command line for the ported default workflow. Flag names and defaults
//! follow tedana; a flag whose non-default behaviour is not ported is an error, never ignored.

use meica_core::io::read_nifti;
use meica_core::mapca::Criterion;
use meica_core::output::Writer;
use meica_core::selection::bundled_tree;
use meica_core::workflow::{Options, VERSION, run};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: meica -d FILE [FILE ...] -e TE [TE ...] [--out-dir DIR] [--prefix P]
              [--tree tedana_orig|meica|minimal|demo_external_*|motion_aroma|motion_f|FILE]
              [--external TSV] [--tedpca aic|kic|mdl] [--seed N] [--maxit N] [--maxrestart N]
              [--n-threads N] [--no-reports] [--quiet]

Multi-echo ICA denoising: tedana's default workflow (e07066c) in Rust.
TEs are seconds (< 1) or milliseconds. Outputs use tedana's BIDS names.
Accepted at their tedana defaults only: --convention bids, --masktype dropout,
--fittype loglin, --combmode t2s, --ica-method fastica, --backend cpu, --gscontrol none,
--dummy-scans 0, --n-robust-runs 30. --external takes regressors for trees with an
external_regressor_config; --no-reports only drops their correlation columns (no figures are
made). Accepted and ignored: --png-cmap, --overwrite/-f, --debug.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args).and_then(|(inputs, out_dir, opts, quiet)| {
        let start = std::time::Instant::now();
        let mut log = |m: &str| {
            if !quiet {
                eprintln!("[{:6.1}s] {m}", start.elapsed().as_secs_f64());
            }
        };
        log("Loading input data");
        let echoes = inputs
            .iter()
            .map(|p| read_nifti(p))
            .collect::<Result<Vec<_>, _>>()?;
        let writer = Writer::to_dir(&out_dir)?;
        let names: Vec<String> = inputs.iter().map(|p| relative(&out_dir, p)).collect();
        let (s, _) = run(echoes, &names, writer, &opts, &mut log)?;
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

fn parse(args: &[String]) -> Result<(Vec<PathBuf>, PathBuf, Options, bool), String> {
    let mut opts = Options {
        command: std::iter::once("meica".to_string())
            .chain(args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" "),
        ..Options::default()
    };
    let (mut inputs, mut out_dir, mut quiet) = (Vec::new(), PathBuf::from("."), false);
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
            "--no-reports" => {
                opts.no_reports = true;
                i += 1;
            }
            "--overwrite" | "-f" | "--debug" => i += 1,
            "--png-cmap" => {
                one(&mut i, flag)?;
            }
            "--convention" | "--masktype" | "--fittype" | "--combmode" | "--ica-method"
            | "--backend" | "--gscontrol" | "--dummy-scans" | "--n-robust-runs" => {
                let default = match flag {
                    "--convention" => "bids",
                    "--masktype" => "dropout",
                    "--fittype" => "loglin",
                    "--combmode" => "t2s",
                    "--ica-method" => "fastica",
                    "--backend" => "cpu",
                    "--gscontrol" => "none",
                    "--dummy-scans" => "0",
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
    Ok((inputs, out_dir, opts, quiet))
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
