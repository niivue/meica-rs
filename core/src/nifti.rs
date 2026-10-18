//! The NIfTI-1 header fields meica reads and writes back (NIfTI-1 spec offsets), and the scalar
//! voxel types it decodes. Either byte order; `io` swaps big-endian voxels once on load.

pub struct Header {
    pub dim: [i16; 8],
    pub datatype: i16,
    pub pixdim: [f32; 8],
    pub vox_offset: f32,
    pub scl_slope: f32,
    pub scl_inter: f32,
    pub qform_code: i16,
    pub sform_code: i16,
    pub quaternion: [f32; 3],
    pub qoffset: [f32; 3],
    pub srow: [[f32; 4]; 3],
    pub little_endian: bool,
}

/// One little-endian voxel to f64.
pub type Decode = fn(&[u8]) -> f64;

impl Header {
    /// The first 348 bytes; the byte order is the one in which sizeof_hdr reads 348.
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        let le = match (
            b[..4].try_into().map(i32::from_le_bytes),
            b[..4].try_into().map(i32::from_be_bytes),
        ) {
            (Ok(348), _) => true,
            (_, Ok(348)) => false,
            _ => return Err("not a NIfTI-1 file".into()),
        };
        let bytes = |o: usize, n: usize| -> Vec<u8> {
            let mut v = b[o..o + n].to_vec();
            if !le {
                v.reverse();
            }
            v
        };
        let i16at = |o| i16::from_le_bytes(bytes(o, 2).try_into().unwrap());
        let f32at = |o| f32::from_le_bytes(bytes(o, 4).try_into().unwrap());
        Ok(Header {
            dim: std::array::from_fn(|i| i16at(40 + 2 * i)),
            datatype: i16at(70),
            pixdim: std::array::from_fn(|i| f32at(76 + 4 * i)),
            vox_offset: f32at(108),
            scl_slope: f32at(112),
            scl_inter: f32at(116),
            qform_code: i16at(252),
            sform_code: i16at(254),
            quaternion: std::array::from_fn(|i| f32at(256 + 4 * i)),
            qoffset: std::array::from_fn(|i| f32at(268 + 4 * i)),
            srow: std::array::from_fn(|r| std::array::from_fn(|k| f32at(280 + 16 * r + 4 * k))),
            little_endian: le,
        })
    }

    /// nibabel `get_best_affine` (rows 0..3): sform, else qform, else the centred voxel-size
    /// affine with x flipped.
    pub fn affine(&self) -> [[f32; 4]; 3] {
        let zoom = |a: usize| self.pixdim[a] as f64;
        if self.sform_code > 0 {
            return self.srow;
        }
        let mut m = [[0f64; 4]; 3];
        if self.qform_code > 0 {
            let [b, c, d] = self.quaternion.map(|q| q as f64);
            // nibabel fillpositive: a rounding-negative w² is 0.
            let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
            let s = 2.0 / (a * a + b * b + c * c + d * d);
            let r = [
                [
                    1.0 - s * (c * c + d * d),
                    s * (b * c - a * d),
                    s * (b * d + a * c),
                ],
                [
                    s * (b * c + a * d),
                    1.0 - s * (b * b + d * d),
                    s * (c * d - a * b),
                ],
                [
                    s * (b * d - a * c),
                    s * (c * d + a * b),
                    1.0 - s * (b * b + c * c),
                ],
            ];
            let qfac = if self.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
            let vox = [zoom(1), zoom(2), zoom(3) * qfac];
            for (i, row) in m.iter_mut().enumerate() {
                for k in 0..3 {
                    row[k] = r[i][k] * vox[k];
                }
                row[3] = self.qoffset[i] as f64;
            }
        } else {
            for (i, row) in m.iter_mut().enumerate() {
                let z = if i == 0 { -zoom(1) } else { zoom(i + 1) };
                row[i] = z;
                row[3] = -(self.dim[i + 1] as f64 - 1.0) / 2.0 * z;
            }
        }
        m.map(|row| row.map(|x| x as f32))
    }

    /// Bytes per voxel and its decoder, for the scalar types.
    pub fn decoder(&self) -> Result<(usize, Decode), String> {
        macro_rules! le {
            ($t:ty) => {
                |c: &[u8]| <$t>::from_le_bytes(c.try_into().unwrap()) as f64
            };
        }
        Ok(match self.datatype {
            2 => (1, |c| c[0] as f64),
            256 => (1, |c| c[0] as i8 as f64),
            4 => (2, le!(i16)),
            512 => (2, le!(u16)),
            8 => (4, le!(i32)),
            768 => (4, le!(u32)),
            16 => (4, le!(f32)),
            64 => (8, le!(f64)),
            t => {
                return Err(format!(
                    "Unsupported NIfTI datatype {t}. Use a scalar intensity image."
                ));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Header;

    #[test]
    fn best_affine_matches_nibabel() {
        // nibabel get_best_affine for this header, qform-only then with no codes.
        let mut h = Header {
            dim: [4, 4, 5, 6, 7, 1, 1, 1],
            datatype: 16,
            pixdim: [-1.0, 2.0, 3.0, 4.0, 1.5, 1.0, 1.0, 1.0],
            vox_offset: 352.0,
            scl_slope: 1.0,
            scl_inter: 0.0,
            qform_code: 1,
            sform_code: 0,
            quaternion: [0.1, 0.2, 0.3],
            qoffset: [10.0, 20.0, 30.0],
            srow: [[0.0; 4]; 3],
            little_endian: true,
        };
        let close = |a: [[f32; 4]; 3], b: [[f32; 4]; 3]| {
            a.iter()
                .flatten()
                .zip(b.iter().flatten())
                .all(|(x, y)| (x - y).abs() < 1e-5)
        };
        assert!(close(
            h.affine(),
            [
                [1.48, -1.5492514, -1.723779, 10.0],
                [1.1928343, 2.4, 0.26188946, 20.0],
                [-0.6218895, 0.9164171, -3.6, 30.0],
            ]
        ));
        h.qform_code = 0;
        assert!(close(
            h.affine(),
            [
                [-2.0, 0.0, 0.0, 3.0],
                [0.0, 3.0, 0.0, -6.0],
                [0.0, 0.0, 4.0, -10.0]
            ]
        ));
    }
}
