//! Browser bindings: add echoes, run tedana's default workflow, read back the output files.

use meica_core::io::{Nifti, parse_nifti};
use meica_core::json::Json;
use meica_core::mapca::Criterion;
use meica_core::moco;
use meica_core::output::Writer;
use meica_core::workflow::{Options, run};
use wasm_bindgen::prelude::*;

// Async `initThreadPool(n)` for the JS glue; call it once before `run`.
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn report_panic(message: &str);
    /// Progress messages go to `globalThis.meicaProgress`, which the wrapper defines.
    #[wasm_bindgen(js_namespace = globalThis, js_name = meicaProgress)]
    fn progress(message: &str);
}

#[wasm_bindgen(start)]
pub fn initialize_diagnostics() {
    std::panic::set_hook(Box::new(|info| report_panic(&info.to_string())));
}

#[wasm_bindgen]
#[derive(Default)]
pub struct Session {
    echoes: Vec<Nifti>,
    names: Vec<String>,
    mask: Option<Vec<u8>>,
    files: Vec<(String, Vec<u8>)>,
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Session {
        Session::default()
    }

    /// Add the next echo (NIfTI or NIfTI-gz bytes, ascending TE order).
    pub fn add_echo(&mut self, name: &str, bytes: Vec<u8>) -> Result<(), JsValue> {
        let image = parse_nifti(bytes).map_err(|e| JsValue::from_str(&format!("{name}: {e}")))?;
        self.echoes.push(image);
        self.names.push(name.into());
        Ok(())
    }

    /// tedana --mask for the next run: NIfTI or NIfTI-gz bytes on the echoes' grid.
    pub fn set_mask(&mut self, bytes: Vec<u8>) {
        self.mask = Some(bytes);
    }

    /// Run with JSON options: `tes` (required), `tree`, `tedpca`, `seed`, `maxit`,
    /// `maxrestart`, `prefix`, `command`, `external` (regressors TSV text), `dummy_scans`, named
    /// as tedana's flags. Returns a JSON summary.
    pub fn run(&mut self, options: &str) -> Result<String, JsValue> {
        let fail = |e: String| JsValue::from_str(&e);
        let o = Json::parse(options).map_err(fail)?;
        let d = Options::default();
        // As the CLI: absent options take defaults, present ones must have the right type.
        let text = |k: &str, d: &str| match o.get(k) {
            None => Ok(d.to_string()),
            Some(v) => v
                .as_str()
                .map(String::from)
                .ok_or_else(|| fail(format!("options.{k} must be a string"))),
        };
        let count = |k: &str, d: u64, max: u64| match o.get(k) {
            None => Ok(d),
            Some(&Json::Int(i)) if u64::try_from(i).is_ok_and(|v| v <= max) => Ok(i as u64),
            _ => Err(fail(format!("options.{k} must be an integer in 0..={max}"))),
        };
        let opts = Options {
            tes: o
                .get("tes")
                .and_then(Json::f64s)
                .ok_or_else(|| fail("options.tes is required".into()))?,
            tree: text("tree", &d.tree)?,
            criterion: match o.get("tedpca") {
                Some(_) => Criterion::parse(&text("tedpca", "")?).map_err(fail)?,
                None => d.criterion,
            },
            seed: count("seed", d.seed.into(), u32::MAX.into())? as u32,
            maxit: count("maxit", d.maxit as u64, usize::MAX as u64)? as usize,
            maxrestart: count("maxrestart", d.maxrestart as u64, usize::MAX as u64)? as usize,
            prefix: text("prefix", "")?,
            command: text("command", "meica (browser)")?,
            external: o
                .get("external")
                .map(|_| text("external", ""))
                .transpose()?,
            no_reports: false,
            dummy_scans: count("dummy_scans", 0, usize::MAX as u64)? as usize,
            mask: self.mask.take(),
        };
        let echoes = std::mem::take(&mut self.echoes);
        let names = std::mem::take(&mut self.names);
        let (s, files) = run(echoes, &names, Writer::in_memory(), &opts, &mut |m| {
            progress(m)
        })
        .map_err(fail)?;
        self.files = files;
        Ok(Json::Obj(vec![
            ("n_components".into(), Json::Int(s.n_components as i64)),
            ("accepted".into(), Json::Int(s.accepted as i64)),
            ("rejected".into(), Json::Int(s.rejected as i64)),
            ("seed".into(), Json::Int(s.seed as i64)),
            ("n_iter".into(), Json::Int(s.n_iter as i64)),
        ])
        .to_string())
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn file_name(&self, index: usize) -> String {
        self.files[index].0.clone()
    }

    /// Move a file's bytes out (the session keeps only its name), so memory is released as the
    /// page saves files.
    pub fn take_file(&mut self, index: usize) -> Vec<u8> {
        std::mem::take(&mut self.files[index].1)
    }
}

// `--moco` command lines (core/src/moco.rs) for js `prepareMoco`, so JavaScript never spells a flag.

#[wasm_bindgen]
pub fn moco_estimate(echo1: &str, dir: &str) -> Vec<String> {
    moco::estimate(echo1, dir)
}

#[wasm_bindgen]
pub fn moco_mask(dir: &str) -> Vec<String> {
    moco::mask(dir)
}

/// `with_confounds` for echo 1: also write the confounds (TR in seconds).
#[wasm_bindgen]
pub fn moco_apply(
    echo: &str,
    dir: &str,
    out: &str,
    with_confounds: bool,
    tr: f32,
    dummy_scans: usize,
) -> Vec<String> {
    moco::apply(echo, dir, out, with_confounds.then_some((tr, dummy_scans)))
}

/// Repetition time (header pixdim[4]) of NIfTI bytes, for `moco_apply`.
#[wasm_bindgen]
pub fn nifti_tr(bytes: &[u8]) -> Result<f32, String> {
    meica_core::io::header_of(bytes).map(|h| h.pixdim[4])
}

/// The dummy count, mask and confounds files the steps write.
#[wasm_bindgen]
pub fn moco_paths(dir: &str) -> Vec<String> {
    [moco::NSS, moco::MASK, moco::CONFOUNDS]
        .map(|n| moco::path(dir, n))
        .to_vec()
}

#[wasm_bindgen]
pub fn moco_dummy_scans(nss_json: &str) -> Result<usize, String> {
    moco::dummy_scans(nss_json)
}

/// Whether a tree (bundled name or JSON) takes external regressors, as `--moco` passes its
/// confounds only to those.
#[wasm_bindgen]
pub fn tree_takes_regressors(tree: &str) -> Result<bool, String> {
    meica_core::selection::load_tree(tree).map(|t| meica_core::selection::takes_regressors(&t))
}
