//! NIfTI-1 reading (plain or gzip) with nibabel's slope/intercept rules. Voxels stay in the
//! file's own type and are decoded on access, so an int16 series costs 2 bytes per value.

use crate::nifti::{Decode, Header};
use flate2::read::MultiGzDecoder;
use std::io::Read;
use std::path::Path;

pub struct Nifti {
    pub header: Header,
    /// x, y, z, t (missing dimensions are 1).
    pub dims: [usize; 4],
    raw: Vec<u8>,
    offset: usize,
    width: usize,
    decode: Decode,
    slope: f64,
    inter: f64,
}

impl Nifti {
    /// Scaled value of voxel `i` in NIfTI order (x fastest, volumes slowest).
    pub fn value(&self, i: usize) -> f64 {
        let o = self.offset + i * self.width;
        let mut v = (self.decode)(&self.raw[o..o + self.width]);
        // nibabel skips identity scaling, so values stay bit-exact.
        if self.slope != 1.0 {
            v *= self.slope;
        }
        if self.inter != 0.0 {
            v += self.inter;
        }
        v
    }
}

/// A NIfTI-1 image from file bytes (gzip or plain).
pub fn parse_nifti(bytes: Vec<u8>) -> Result<Nifti, String> {
    let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
        // Size from the gzip trailer (length mod 2^32) so read_to_end does not overshoot by up
        // to 2x; capped at deflate's maximum ratio since the header is untrusted.
        let n = bytes.len();
        let isize = u32::from_le_bytes(bytes[n.saturating_sub(4)..].try_into().unwrap_or([0; 4]));
        // A hint only: past isize::MAX (2 GiB on wasm32) with_capacity would panic. read_to_end
        // itself reports an allocation failure as an error.
        let mut out = Vec::new();
        let _ = out.try_reserve((isize as usize).min(n.saturating_mul(1032)));
        MultiGzDecoder::new(&bytes[..])
            .read_to_end(&mut out)
            .map_err(|e| e.to_string())?;
        out
    } else {
        bytes
    };
    if bytes.len() < 352 {
        return Err("not a NIfTI-1 file".into());
    }
    let header = Header::parse(&bytes[..348])?;
    let nd = header.dim[0].clamp(0, 7) as usize;
    if header.dim[5..=nd.max(4)].iter().any(|&d| d > 1) {
        return Err("5D and higher images are not supported".into());
    }
    let dims: [usize; 4] = std::array::from_fn(|a| {
        if a < nd {
            header.dim[a + 1].max(1) as usize
        } else {
            1
        }
    });
    let (width, decode) = header.decoder()?;
    if !(header.vox_offset.is_finite() && header.vox_offset >= 352.0) {
        return Err("bad vox_offset".into());
    }
    let offset = header.vox_offset as usize;
    // Checked: on wasm32 a large image's byte count overflows usize.
    let end = dims
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
        .and_then(|n| n.checked_add(offset));
    if end.is_none_or(|end| bytes.len() < end) {
        return Err("truncated image data".into());
    }
    let mut bytes = bytes;
    if !header.little_endian && width > 1 {
        bytes[offset..end.unwrap()]
            .chunks_exact_mut(width)
            .for_each(<[u8]>::reverse);
    }
    // nibabel: a zero or non-finite slope means unscaled.
    let (slope, inter) = match header.scl_slope as f64 {
        s if s.is_finite() && s != 0.0 => (
            s,
            if header.scl_inter.is_finite() {
                header.scl_inter as f64
            } else {
                0.0
            },
        ),
        _ => (1.0, 0.0),
    };
    Ok(Nifti {
        header,
        dims,
        raw: bytes,
        offset,
        width,
        decode,
        slope,
        inter,
    })
}

pub fn read_nifti(path: &Path) -> Result<Nifti, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_nifti(bytes).map_err(|e| format!("{}: {e}", path.display()))
}
