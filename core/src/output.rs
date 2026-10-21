//! Writers for tedana's outputs: NIfTI-1 images as nilearn/nibabel save them, pandas-style TSV
//! and Python-style JSON, plus the registry of written files.

use crate::json::{Json, py_float};
use crate::nifti::Header;
use flate2::{Compression, write::GzEncoder};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Output files as (name, bytes).
pub type Files = Vec<(String, Vec<u8>)>;

/// Image values: float32 maps or the int32 adaptive mask.
pub enum Voxels<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
}

/// One TSV column.
pub enum Column {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Text(Vec<String>),
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Column::Float(v) => v.len(),
            Column::Int(v) => v.len(),
            Column::Text(v) => v.len(),
        }
    }
}

pub struct Writer {
    /// Written files when no directory was given (the browser), in write order.
    pub files: Files,
    dir: Option<PathBuf>,
    /// Geometry of the inputs (sform, quaternion, voxel sizes, TR).
    template: Option<Header>,
    /// (description, file name) in write order, for desc-tedana_registry.json.
    registry: Vec<(String, Json)>,
}

impl Writer {
    /// Write each file into `dir` as it is produced.
    pub fn to_dir(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Writer {
            dir: Some(dir.to_path_buf()),
            ..Writer::in_memory()
        })
    }

    /// Keep every file in `files`.
    pub fn in_memory() -> Self {
        Writer {
            files: Vec::new(),
            dir: None,
            template: None,
            registry: Vec::new(),
        }
    }

    /// The input geometry images take theirs from.
    pub fn set_template(&mut self, header: Header) {
        self.template = Some(header);
    }

    fn put(&mut self, description: &str, name: &str, bytes: Vec<u8>) -> Result<(), String> {
        match &self.dir {
            Some(dir) => {
                let path = dir.join(name);
                std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            }
            None => self.files.push((name.into(), bytes)),
        }
        if !description.is_empty() {
            self.registry.retain(|(d, _)| d != description);
            self.registry
                .push((description.into(), Json::Str(name.into())));
        }
        Ok(())
    }

    /// A gzipped NIfTI-1 image of `dims` (x, y, z, volumes), values in NIfTI order. As nibabel
    /// writes a new image: sform code 2 from the input's best affine, qform code 0, no scaling, and
    /// pixdim[4] the input TR for 4D images.
    pub fn image(
        &mut self,
        description: &str,
        name: &str,
        dims: &[usize],
        v: Voxels,
    ) -> Result<(), String> {
        let t = self
            .template
            .as_ref()
            .ok_or("image written before the input geometry was set")?;
        let mut h = vec![0u8; 352];
        let put_i16 =
            |h: &mut [u8], o: usize, x: i16| h[o..o + 2].copy_from_slice(&x.to_le_bytes());
        let put_f32 =
            |h: &mut [u8], o: usize, x: f32| h[o..o + 4].copy_from_slice(&x.to_le_bytes());
        h[0..4].copy_from_slice(&348i32.to_le_bytes());
        let ndim = dims.len() as i16;
        put_i16(&mut h, 40, ndim);
        for a in 0..7 {
            put_i16(&mut h, 42 + 2 * a, dims.get(a).map_or(1, |&n| n as i16));
        }
        let (code, bitpix) = match v {
            Voxels::F32(_) => (16, 32),
            Voxels::I32(_) => (8, 32),
        };
        put_i16(&mut h, 70, code);
        put_i16(&mut h, 72, bitpix);
        put_f32(&mut h, 76, t.pixdim[0]);
        for a in 1..4 {
            put_f32(&mut h, 76 + 4 * a, t.pixdim[a]);
        }
        put_f32(&mut h, 92, if ndim == 4 { t.pixdim[4] } else { 1.0 });
        for a in 5..8 {
            put_f32(&mut h, 76 + 4 * a, 1.0);
        }
        put_f32(&mut h, 108, 352.0);
        put_f32(&mut h, 112, f32::NAN);
        put_f32(&mut h, 116, f32::NAN);
        put_i16(&mut h, 252, 0);
        put_i16(&mut h, 254, 2);
        for a in 0..3 {
            put_f32(&mut h, 256 + 4 * a, t.quaternion[a]);
            put_f32(&mut h, 268 + 4 * a, t.qoffset[a]);
        }
        for (r, row) in t.affine().iter().enumerate() {
            for (k, &x) in row.iter().enumerate() {
                put_f32(&mut h, 280 + (r * 4 + k) * 4, x);
            }
        }
        h[344..348].copy_from_slice(b"n+1\0");
        let mut gz = GzEncoder::new(Vec::new(), Compression::new(1)); // nibabel's default level
        gz.write_all(&h).map_err(|e| e.to_string())?;
        // Stream the voxels in blocks rather than holding a second, uncompressed copy.
        let mut block = Vec::with_capacity(1 << 16);
        let mut put = |x: [u8; 4]| -> std::io::Result<()> {
            block.extend_from_slice(&x);
            if block.len() >= 1 << 16 {
                gz.write_all(&block)?;
                block.clear();
            }
            Ok(())
        };
        match v {
            Voxels::F32(d) => d.iter().try_for_each(|x| put(x.to_le_bytes())),
            Voxels::I32(d) => d.iter().try_for_each(|x| put(x.to_le_bytes())),
        }
        .and_then(|_| gz.write_all(&block))
        .map_err(|e| e.to_string())?;
        let bytes = gz.finish().map_err(|e| e.to_string())?;
        self.put(description, name, bytes)
    }

    /// pandas `to_csv(sep="\t", na_rep="n/a", index=False)` after tedana's blank-to-NaN.
    pub fn tsv(
        &mut self,
        description: &str,
        name: &str,
        columns: &[(String, Column)],
    ) -> Result<(), String> {
        let rows = columns.first().map_or(0, |(_, c)| c.len());
        // pandas QUOTE_MINIMAL: names and tags come from the tree and the regressors file.
        let quote = |s: &str| {
            if s.contains(['\t', '\n', '\r', '"']) {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.to_string()
            }
        };
        let mut out = columns
            .iter()
            .map(|(n, _)| quote(n))
            .collect::<Vec<_>>()
            .join("\t");
        out.push('\n');
        for r in 0..rows {
            let cells: Vec<String> = columns
                .iter()
                .map(|(_, c)| match c {
                    Column::Float(v) if v[r].is_nan() => "n/a".into(),
                    Column::Float(v) => py_float(v[r]),
                    Column::Int(v) => v[r].to_string(),
                    Column::Text(v) if v[r].is_empty() => "n/a".into(),
                    Column::Text(v) => quote(&v[r]),
                })
                .collect();
            out.push_str(&cells.join("\t"));
            out.push('\n');
        }
        self.put(description, name, out.into_bytes())
    }

    /// `json.dump(indent=4, sort_keys=True)`.
    pub fn json(&mut self, description: &str, name: &str, value: &Json) -> Result<(), String> {
        self.put(description, name, value.pretty().into_bytes())
    }

    /// The registry `name` (the inputs as the caller names them) and the kept files.
    pub fn finish(mut self, name: &str, input_names: &[String]) -> Result<Files, String> {
        let mut registry = self.registry.clone();
        registry.push(("input img".into(), Json::strings(input_names)));
        self.put("", name, Json::Obj(registry).pretty().into_bytes())?;
        Ok(self.files)
    }
}

/// Gzip at nibabel's default level.
pub fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut gz = GzEncoder::new(Vec::new(), Compression::new(1));
    gz.write_all(bytes)
        .and_then(|_| gz.finish())
        .expect("gzip to memory")
}

/// tedana `add_decomp_prefix`: zero-padded to the digits of `max_value`.
pub fn component_name(prefix: &str, index: usize, max_value: usize) -> String {
    let digits = max_value.max(1).ilog10() as usize + 1;
    format!("{prefix}_{index:0digits$}")
}
