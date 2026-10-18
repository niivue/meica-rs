//! Rust port of tedana's multi-echo ICA pipeline.

pub mod combine;
pub mod decay;
pub mod external;
pub mod fastica;
pub mod fft;
pub mod io;
pub mod json;
pub mod linalg;
pub mod mapca;
pub mod mask;
pub mod math;
pub mod metrics;
pub mod nifti;
pub mod output;
pub mod pipeline;
pub mod rng;
pub mod selection;
pub mod stats;
pub mod workflow;

/// Size the global thread pool (no-op without the `parallel` feature).
pub fn set_threads(n: usize) -> Result<(), String> {
    #[cfg(feature = "parallel")]
    return rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| e.to_string());
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        Ok(())
    }
}
