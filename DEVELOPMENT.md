# Developing meica

Maintainer notes: how meica is tested against tedana, the numerics that make it match, and the
traps found on the way. Gates only tighten; regenerate references rather than loosening a
tolerance.

```sh
make reference MEICA_HF_DIR=DIR   # pinned HF download; MEICA_REFERENCE_DIR=DIR/me-ica/motor-3echo[-moco]
make test        # cargo test --workspace --release
make test-real   # also the pipeline stages against tedana's outputs (needs MEICA_REFERENCE_DIR)
make check-outputs PYTHON=python   # CLI on the example, every file against tedana's
make check-external PYTHON=python  # the five regressor trees against tedana's (motor-3echo-moco)
make wasm        # js/dist/threaded (nightly-2025-11-15 build-std; not published) and js/dist/single (stable)
make wasm-parity # single-threaded WebAssembly in Node: every output byte-identical to native
make lint        # clippy -D warnings (native and wasm32), fmt --check
make fixtures PYTHON=python   # regenerate core/tests/fixtures (needs numpy, scipy, mpmath)
```

Options that select unported tedana code (curvefit, robustica, masks, tedort, gscontrol, dummy
scans, verbose outputs) are errors. Outputs are tedana's BIDS files except the HTML report,
figures, `report.txt` and `references.bib`. The core depends on `flate2` (gzip) and, behind the
default `parallel` feature, `rayon`; `core/src/nifti.rs` is its own NIfTI-1 header reader.
Outputs carry nibabel's `get_best_affine` (`Header::affine`: sform, else qform, else a centred
x-flipped voxel affine) as sform 2, qform 0; sform inputs keep their srow bytes. Untrusted input
(NIfTI, JSON, TSV, options) is validated up front and errors rather than panics; regressions
in `tests/primitives.rs` (`bad_inputs_are_errors_not_panics`).
Fixtures come from `validation/gen_fixtures.py` against tedana e07066c's environment: Python
3.13.2, numpy 2.3.4, scipy 1.16.3, mpmath 1.3.0, on macOS arm64. They are deterministic.
`validation/deid_example.py` built the example from the raw scan, `gen_stage_reference.py` the
`stages/` files, `gen_external_regressors.py` the demo trees' `regressors.tsv`.

## Reference data

`MEICA_REFERENCE_DIR` holds `example/` (the published 3-echo example), `reference/` (tedana
e07066c's outputs on it, default options) and `stages/` (`validation/gen_stage_reference.py
EXAMPLE_DIR OUT_DIR`, from tedana's and mapca's own code: the EPI mask, maPCA's decisions and
spectra, the f64 ICA input `kept_data`, 46 MB, and FastICA's results). All three are on Hugging
Face: dataset `neurodeskorg/webapps`, revision 121b25dde5caffc8f880eec87cca29b4e569b7a9,
`me-ica/motor-3echo/{example,reference,stages}`. On it, the EPI mask, adaptive mask, T2*, S0
and optimal combination are bit-identical to tedana's float32 outputs; so are the denoised
series and the adaptive mask. `validation/compare_outputs.py` holds all 27 files to 1e-6 of each
file's range (finite values; NaN/inf must match exactly, tested by `test_compare_outputs.py`); the worst is 1.5e-8 (ICA z-maps), tables agree to 1e-12, the status table and
classifications (7 accepted, 6 rejected) exactly. One-off runs with `--tree meica` (7/6) and `--tree minimal`
(2/11) agreed with tedana's on every file as well. Every maPCA decision is
exact (IID estimates, depth 2, AIC/KIC/MDL 13/11/10); criterion curves agree to 3e-13 and
kept_data to 2e-13 (LAPACK, BLAS and pocketfft are in Python's path). One-off runs on the
untrimmed source scan (uint16, 150 volumes) matched as well: bit-identical through optcom,
maPCA 15/14/11 with 122/136 components at 90/95% variance, FastICA seed 42 converging in 61
iterations, 9 accepted / 6 rejected, and every output file within tolerance (worst 1e-7). On the example FastICA converges in 35 iterations, as sklearn does, and tedana's
z-scored mixing matrix agrees to 2.4e-12.

Timings are in README "Benchmarks" (native threads give 1.7-1.9x; one thread is slower than
tedana `--no-reports`, whose BLAS is threaded). maPCA is the slowest stage; its autocorrelation
FFTs run Bluestein on odd 2d-1 lengths (padding to a power of two would be ~0.5 s faster but not
byte-identical). The regressor fits add no measurable time. The full scan is not published
(maintainer copy).

## External regressors

`core/src/external.rs` ports `metrics/external.py` and `frequency.py`: F tests of the mixing on
the full and partial regressor models (`Fstat`/`pval`/`R2stat` columns), AROMA's `max_rp_corr`
and `HFC`, and the "external regressor correlation" columns tedana's report step appends
(dropped by `--no-reports`, as in tedana). Regressor names use Python `re.match` (IGNORECASE);
`py_match` covers literals, classes, groups, lookaheads, alternation and quantifiers and errors on
the rest. `max_rp_corr`'s 1000 subsamples use numpy's `default_rng(ICA seed)`: `rng::Pcg64`
reproduces SeedSequence, PCG64 and `Generator.choice` draw for draw. Against tedana on the
motion-corrected example (100 volumes) and the full scan (150, which adds a linear detrend term):
every file agrees; worst metric difference 3e-11 (F), HFC exact. tedana builds its F designs by
iterating a Python `set`, so its own column order (and last bits) varies with `PYTHONHASHSEED`;
Rust orders regressors as the sorted expanded names. WebAssembly is byte-identical to native.
`linalg::lstsq` is QR at full rank and numpy's minimum-norm SVD (rcond = eps·max(m, n)) when
rank-deficient; F degrees of freedom still count columns, as tedana. Every model needs residual
df (detrend rank + regressors < volumes), and F-model columns that are non-finite or constant
after the NaN fill are rejected by name (tedana errors on NaN/inf, and usually on constants,
though rounding lets some through: rejecting all is deliberate). HFC errors up front when no FFT bin lies above
0.01 Hz (TR not in seconds).

`core/trees/motion_aroma.json` and `motion_f.json` are tedana's demo AROMA and motion/task trees
with their patterns pointed at fMRIPrep confound names (`^(trans|rot)_[xyz]$`, plus derivatives and
squares for the F test; no CSF or task model). tedana runs them unchanged from the HF copies
(`--tree trees/motion_aroma.json --external example/…_desc-confounds_timeseries.tsv`), so the
same data regression-tests both; `check-external` also requires the bundled and HF copies to be
identical.

## WebAssembly

`wasm/` exposes a `Session` (add echoes as bytes, `run` with tedana's option names as JSON,
take the output files back); `js/` (`@niivue/meica`) wraps it. The threaded build uses
wasm-bindgen-rayon behind the default `threads` feature, so the page must be cross-origin
isolated; memory may grow to 4 GiB (one-off WebAssembly runs on the 150-volume full scan peaked at 1.1-1.4 GB; not re-measured). The NIfTI reader keeps
each input in its own type (2 bytes per int16 voxel) and decodes on access. `make wasm-parity`
holds all 27 example outputs byte-identical between native and single-threaded WebAssembly; a
one-off run of the threaded build in headless Chromium (8 threads, 8.6 s) was byte-identical
too, as was the 150-volume scan in Node (19 s, 9 accepted / 6 rejected). Absent `run` options
take `Options::default`; present ones of the wrong type or range are errors. `loadMeica` caches
one promise per build URL, so the thread pool starts once, and falls back to `single/` when
`threaded/` is not served or fails to start. npm ships `dist/single` only (release.yml fails on
`dist/threaded`) until the licence is resolved (AGENTS.md). `build-wasm.mjs` remaps `$HOME` out
of panic paths.

## Numerics

- f64, except where tedana itself is f32: it loads data as float32, so the adaptive mask's
  temporal means are f32 pairwise sums and the decay fit sees float32 `log(|x| + 1)`. Sums
  reproduce numpy's pairwise summation. Every reduction runs in a fixed order; nothing depends
  on thread count.
- TEs are in seconds internally; millisecond input is divided by 1000 as tedana's
  `check_te_values` does, so T2* maps are in seconds.
- `math.rs` uses only IEEE +, -, *, / and sqrt, so results are identical on every target. A
  one-off check (2026-10-17) hashed 1.2M exp/ln/tanh/normal draws plus FFT, SVD and eigh
  outputs; native arm64, wasm32 and wasm32+simd128 gave the same digest. A committed
  native-vs-WASM test is `make wasm-parity`.
- Bounds, checked by `tests/primitives.rs`: `ln` correctly rounded on all fixture points
  (and on a one-off 200k-point set near the reduction boundaries, where numpy's own log missed
  42); `exp` within 1 ulp; `tanh` within 2 ulp (msun's bound;
  numpy's own tanh is not correctly rounded on 16% of the points); lgamma within 1e-14;
  betainc, t CDF (df <= 1000) and F(1, d) quantiles within 5e-13 of scipy;
  FFT within 1e-14 of peak; SVD and eigh values within 1e-13 of LAPACK.
- `RandomState` reproduces numpy's legacy generator: seeding, `random_sample` and `normal`.
  The tests pin seed 42 at n = 13 and 15 against numpy 2.3.4 on macOS arm64. A one-off check
  also matched all 37,653 draws for seeds 42-52 (the maxrestart range) at n = 2-40. numpy's
  own draws can differ by an ulp on other platforms (unfused r2, a different libm log).

## Traps

- HFC: tedana's TR is the header's float32 and numpy 2 (NEP 50) keeps `1/(n*tr)` and
  `nyquist - 0.01` in float32; computing them in f64 moves HFC by 1e-9.
- The external-regressor references come from tedana runs with reports on; that is what adds the
  correlation columns to `desc-tedana_metrics.tsv`/`.json`.
- numpy's macOS arm64 build fuses `x1*x1 + x2*x2` in the polar gauss into an FMA. The unfused
  sum moves 38 of 225 draws by up to 18 ulp, so `rng.rs` uses `mul_add` there; that is exactly
  rounded on every target.
- scipy's arm64 build fuses the multiply-add in `gaussian_filter1d`'s symmetric kernel loop, so
  `mask::smooth_fwhm1` uses `mul_add` too. Unfused, 25 of 210 smoothed voxels moved by 1-2 ulp.
  These two are the only FMAs; the references are macOS arm64 builds, and an x86 numpy/scipy
  without FMA would differ by an ulp here and there.
- An msun-accurate `ln` (< 1 ulp) still differs from numpy's in 8 of 225 draws. `ln` carries
  the series in double-double through s^5 and rounds once.
- mapca reshapes in Fortran order, so its rows (and tedana's ICA input) are mask voxels in
  NIfTI order, not tedana's C order. Only rounding depends on it, but the port keeps it.
- mapca's `PCA(copy=False)` centres its data in place, so `u_` (and the ICA input) are built
  from the centred matrix. Using the uncentred one is off by O(1).
- The entropy-rate autocorrelation runs on the 2d - 1 grid itself (no wrap-around at that size)
  rather than a power-of-two pad: 10x less memory traffic. fftn lines run in parallel and are
  independent, so results do not depend on the thread count.
- numpy sums pairwise only along a memory-contiguous axis; reductions across rows of a C-ordered
  matrix (column means, `zscore(axis=0)`) are sequential. Each port follows the layout of the
  array numpy reduces.
- The "no BOLD components" ICA restart has only been exercised with a doctored tree (no node
  tags Likely BOLD): seed 42, then 43, then it stops at `--maxrestart`, as tedana's loop does.
- Seed -1 (tedana's random seed) is not supported: the point of the port is reproducible runs.
- Component subsets are ascending; tedana's `list(set(...))` order only matters to
  `kappa_elbow_kundu`, which the bundled trees call on "all".
- Python joins classification tags in set order, which is randomised per process; meica keeps
  insertion order (compare tags as sets).
- `trees/metric_metadata.json` is tedana's `get_metadata` text, generated once from tedana.
- `betainc` loses about 1e-12 when a >> b near x = (a+1)/(a+b+2): a t CDF at df ~ 1e5. Only
  the reported `signal-noise_p` metric reaches that regime.
- Thin SVD is Householder QR plus one-sided Jacobi on R. A zero singular value leaves a zero U
  column. Signs are raw; callers apply sklearn's `svd_flip` and `u *= sign(u[0])`.
- TSV cells use pandas' QUOTE_MINIMAL quoting: names and tags come from trees and regressor
  files.
- `--tree NAME`: a bundled tree wins over a same-named file, as in tedana.

## Open items

- `validation/wasm_parity.mjs` duplicates the js loader; it could call `loadMeica`/`runMeica`.
- The nightly toolchain is pinned in three places: `ci.yml`, `release.yml`, `js/scripts/build-wasm.mjs`.
- WebAssembly ignores `no_reports` (always adds the report columns).
- `rng` `choice` with n > 10000 is not fixture-verified.
- `selection.rs`: four `comps.is_empty()` blocks repeat `set_cross` + push; `publish` covers them
  (re-run `check-outputs check-external` after).
- The threaded loader cache is untested in a browser.
- Neurodesk webapp: `copy-runtime.mjs` deploys the threaded build (AGENTS.md); `src/worker.js`
  reports "Starting N CPU workers" even after falling back to single; the e2e spec header says it
  tests the threaded worker.
- Neurodesk `expected.json` hashes not re-checked after the audit (default outputs are
  byte-identical, so they should hold).
