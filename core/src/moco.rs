//! `--moco`: the niimath and brainchop-mindgrab command lines that prepare the echoes. meica never
//! links either tool; the CLI runs them as processes and the browser as WebAssembly (fastprep's
//! drivers), both in a working folder `dir` (absolute in the browser, where the driver stages only
//! `/` paths). niimath names images by `FSLOUTPUTTYPE`, not by the extension asked for, and the
//! browser driver uses NIFTI_GZ, so the CLI does too and the names below end in `.nii.gz`.
//!
//! 1. `estimate`: echo 1 -> fMRIPrep's BOLD reference, dummy count (`nss.json`), motion (`hmc.1D`).
//! 2. `mask`: brainchop-mindgrab on the reference (echo 1 has the most signal).
//! 3. `apply`: each echo resampled once through the same motion onto the reference grid, to `out`
//!    (`-`, uncompressed stdout, natively); echo 1 also writes fMRIPrep-named confounds
//!    (`confounds.tsv`) as a side output.

use crate::json::Json;

pub const REFERENCE: &str = "ref.nii.gz";
pub const NSS: &str = "nss.json";
pub const MOTION: &str = "hmc.1D";
pub const MASK: &str = "mask.nii.gz";
pub const CONFOUNDS: &str = "confounds.tsv";

pub fn path(dir: &str, name: &str) -> String {
    format!("{dir}/{name}")
}

pub fn estimate(echo1: &str, dir: &str) -> Vec<String> {
    let (r, n, m) = (path(dir, REFERENCE), path(dir, NSS), path(dir, MOTION));
    let argv = [
        "niimath",
        echo1,
        "-boldref",
        &r,
        &n,
        "-moco",
        "-ref",
        &r,
        "-estimate",
        &m,
    ];
    argv.map(String::from).to_vec()
}

pub fn mask(dir: &str) -> Vec<String> {
    let argv = [
        "brainchop-mindgrab",
        &path(dir, REFERENCE),
        "-o",
        &path(dir, "stripped.nii.gz"),
        "--mask",
        &path(dir, MASK),
    ];
    argv.map(String::from).to_vec()
}

/// One echo to `out`; `confounds` = (TR in seconds, dummy scans) for echo 1.
pub fn apply(echo: &str, dir: &str, out: &str, confounds: Option<(f32, usize)>) -> Vec<String> {
    let (r, m) = (path(dir, REFERENCE), path(dir, MOTION));
    let mut v = vec!["niimath".to_string(), echo.into()];
    if let Some((tr, dummy)) = confounds {
        let side = [
            "-confounds",
            &path(dir, CONFOUNDS),
            "-tr",
            &tr.to_string(),
            "-hmc",
            &m,
            "-ref",
            &r,
            "-brain",
            &path(dir, MASK),
            "-dummy",
            &dummy.to_string(),
        ];
        v.extend(side.map(String::from));
    }
    v.extend(["-resample", &r, "-hmc", &m, out].map(String::from));
    v
}

/// `dummy_scans` from niimath's `nss.json`.
pub fn dummy_scans(nss_json: &str) -> Result<usize, String> {
    match Json::parse(nss_json)?.get("dummy_scans") {
        Some(&Json::Int(n)) if n >= 0 => Ok(n as usize),
        _ => Err(format!("{NSS}: no dummy_scans count")),
    }
}

#[cfg(test)]
mod tests {
    // The cross-platform contract: the CLI and the browser drivers run exactly these lines.
    #[test]
    fn command_lines() {
        assert_eq!(
            super::estimate("/m/e1.nii.gz", "/m").join(" "),
            "niimath /m/e1.nii.gz -boldref /m/ref.nii.gz /m/nss.json -moco -ref /m/ref.nii.gz \
             -estimate /m/hmc.1D"
        );
        assert_eq!(
            super::mask("/m").join(" "),
            "brainchop-mindgrab /m/ref.nii.gz -o /m/stripped.nii.gz --mask /m/mask.nii.gz"
        );
        assert_eq!(
            super::apply("/m/e1.nii.gz", "/m", "-", Some((2.0, 3))).join(" "),
            "niimath /m/e1.nii.gz -confounds /m/confounds.tsv -tr 2 -hmc /m/hmc.1D -ref \
             /m/ref.nii.gz -brain /m/mask.nii.gz -dummy 3 -resample /m/ref.nii.gz -hmc /m/hmc.1D -"
        );
        assert_eq!(
            super::apply("/m/e2.nii.gz", "/m", "/m/out-2.nii", None).join(" "),
            "niimath /m/e2.nii.gz -resample /m/ref.nii.gz -hmc /m/hmc.1D /m/out-2.nii"
        );
        assert_eq!(super::dummy_scans(r#"{"dummy_scans": 2}"#), Ok(2));
        assert!(super::dummy_scans("{}").is_err());
    }
}
