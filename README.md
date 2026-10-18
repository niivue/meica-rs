# meica-rs

Multi-echo independent component analysis (ME-ICA) in Rust, as a native executable and as
WebAssembly.

Single-echo ICA must classify components from their spatial and temporal features alone. Multi-echo
ICA, as implemented in [tedana](https://tedana.readthedocs.io/en/stable/), also uses how each
component's amplitude depends on echo time (TE). With three or more echoes, components that scale
with TE (T2*-weighted BOLD signal, scored by κ) separate from those that do not (non-BOLD
artifacts, scored by ρ), so denoising becomes more objective and automatic.

meica ports tedana's workflow (commit e07066c) and its dimensionality estimate,
[mapca](https://github.com/ME-ICA/mapca) 0.0.8, which follows the Matlab
[GIFT](https://github.com/trendscenter/gift) toolbox. It needs neither Python nor Matlab. Results
match tedana's to floating-point rounding: every output file agrees within 1e-6 of its range, and
classifications are identical.

## Compilation

Native executable ([Rust](https://rustup.rs) 1.85 or later):

```bash
git clone https://github.com/niivue/meica-rs
cd meica-rs
cargo build --release   # target/release/meica
```

WebAssembly ([wasm-pack](https://github.com/rustwasm/wasm-pack) 0.13; the threaded build needs
`rustup toolchain install nightly-2025-11-15 --component rust-src`, both need
`rustup target add wasm32-unknown-unknown`):

```bash
make wasm   # js/dist/threaded and js/dist/single; npm @niivue/meica publishes single only
```

## Usage

The flags are tedana's (`meica --help`). This downloads the motion-corrected sample (three echoes,
100 volumes, with fMRIPrep-style motion confounds) from Hugging Face
([`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli): `pip install huggingface_hub`)
and denoises it with tedana's default decision tree, then with a tree that also rejects
motion-related components:

```bash
hf download neurodeskorg/webapps --repo-type dataset \
  --revision 121b25dde5caffc8f880eec87cca29b4e569b7a9 \
  --include "me-ica/motor-3echo-moco/example/*" --include "me-ica/motor-3echo-moco/trees/*" \
  --local-dir data
cd data/me-ica/motor-3echo-moco
E="example/sub-01_task-motor_echo-1_bold.nii.gz example/sub-01_task-motor_echo-2_bold.nii.gz example/sub-01_task-motor_echo-3_bold.nii.gz"
meica -d $E -e 15.2 35.32 55.42 --out-dir out
meica -d $E -e 15.2 35.32 55.42 --tree motion_aroma \
  --external example/sub-01_task-motor_desc-confounds_timeseries.tsv --out-dir out_aroma
```

`out/desc-denoised_bold.nii.gz` is the denoised series; `desc-tedana_metrics.tsv` lists each
component's metrics and classification. Outputs use tedana's BIDS names, without its HTML report
and figures.

The same commands run in tedana; a bundled tree such as `motion_aroma` is given to tedana as a file:

```bash
tedana -d $E -e 15.2 35.32 55.42 --tree trees/motion_aroma.json \
  --external example/sub-01_task-motor_desc-confounds_timeseries.tsv --out-dir tedana_aroma
```

Decision trees: tedana's `tedana_orig` (default), `meica`, `minimal` and its three
`demo_external_*` trees, plus `motion_aroma` and `motion_f`, which apply tedana's demo AROMA and
F-test trees to fMRIPrep's `trans_*`/`rot_*` confound columns. `--tree` also takes a JSON file; a
bundled name wins over a same-named file, as in tedana. Options that select tedana code not
ported here (for example `--fittype curvefit`, `--ica-method robustica`, `--mask`) stop with an
error rather than being ignored.

In JavaScript (browser or Node 22+):

```js
import { loadMeica, runMeica } from "@niivue/meica";
const api = await loadMeica(); // single-threaded (threaded only from a self-built baseUrl)
const { summary, files } = runMeica({ api, echoes, tes: [15.2, 35.32, 55.42] });
```

`echoes` are `{ name, bytes }` NIfTI files in echo order; `files` are the outputs. The
[Neurodesk webapps](https://github.com/neurodesk/webapps) ME-ICA app is a browser demo built on
this package (webapps.neurodesk.org/me-ica once deployed).

## Benchmarks

The goal was a platform-independent tool that gives tedana's results, not speed. Even so, meica
uses about half tedana's memory and, with its default threads, is faster. Wall time and peak
memory on an Apple M4 Pro (14 cores, `/usr/bin/time -l`), with tedana e07066c (Python 3.13), on
the cropped example (51×68×46×100) and the full scan it was cut from (76×76×46×150):

| Data | Tree | meica | meica `--n-threads 1` | tedana `--no-reports` | tedana |
| --- | --- | --- | --- | --- | --- |
| cropped | tedana_orig | 4.6 s, 0.80 GB | 7.8 s, 0.80 GB | 7.4 s, 1.49 GB | 11.0 s, 2.36 GB |
| | motion_aroma | 3.9 s, 0.80 GB | 7.1 s, 0.80 GB | 6.2 s, 1.49 GB | 10.2 s, 2.37 GB |
| | motion_f | 3.8 s, 0.80 GB | 7.1 s, 0.80 GB | 6.2 s, 1.49 GB | 10.2 s, 2.38 GB |
| full | tedana_orig | 6.6 s, 1.18 GB | 12.6 s, 1.17 GB | 10.0 s, 2.28 GB | 14.9 s, 4.08 GB |
| | motion_aroma | 6.7 s, 1.18 GB | 12.7 s, 1.18 GB | 10.0 s, 2.27 GB | 15.4 s, 4.08 GB |
| | motion_f | 6.6 s, 1.18 GB | 12.6 s, 1.18 GB | 9.9 s, 2.28 GB | 15.4 s, 4.09 GB |

Threads roughly halve meica's time (1.7× cropped, 1.9× full) at no memory cost; single-threaded
it is slower than tedana `--no-reports`, whose numpy BLAS is itself multi-threaded. The cropped
rows replicate from the Usage download (on Linux use `/usr/bin/time -v`); the full scan is not
published:

```bash
/usr/bin/time -l meica -d $E -e 15.2 35.32 55.42 --tree motion_aroma \
  --external example/sub-01_task-motor_desc-confounds_timeseries.tsv --out-dir bench_meica
/usr/bin/time -l tedana -d $E -e 15.2 35.32 55.42 --tree trees/motion_aroma.json \
  --external example/sub-01_task-motor_desc-confounds_timeseries.tsv --out-dir bench_tedana
```

`DEVELOPMENT.md` describes the regression tests, which hold every output to tedana's.

## License

meica is currently GPL-2.0-only. It ports tedana (LGPL-2.1) and mapca, whose
[LICENSE](https://github.com/ME-ICA/mapca/blob/main/LICENSE) is GPL-2.0 (its package metadata says
LGPL) and which follows [GIFT](https://github.com/trendscenter/gift/blob/df369a2f73aab06c4fe97a10ffab841320722281/GroupICAT/icatb/toolbox/icasso122/README_license.txt#L175)
(GPL-2.0). The mapca and GIFT developers have indicated that a
[lesser GPL](https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License) release is
acceptable once they have reviewed this code; meica will then be relicensed to LGPL-2.1. Until
then, software that bundles meica, including its WebAssembly, is subject to GPL-2.0. `NOTICE`
lists other attributions.

## Citations

- DuPre, E. M., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L., Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S., Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S., Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M., Whitaker, K., & Handwerker, D. A. (2021). TE-dependent analysis of multi-echo fMRI with tedana. Journal of Open Source Software, 6(66), 3669. [doi:10.21105/joss.03669](https://doi.org/10.21105/joss.03669)
- Kundu, P., Inati, S. J., Evans, J. W., Luh, W. M., & Bandettini, P. A. (2012). Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI. NeuroImage, 60(3), 1759–1770. [doi:10.1016/j.neuroimage.2011.12.028](https://doi.org/10.1016/j.neuroimage.2011.12.028)
- Li, Y. O., Adali, T., & Calhoun, V. D. (2007). Estimating the number of independent components for functional magnetic resonance imaging data. Human Brain Mapping, 28(11), 1251–1266. [doi:10.1002/hbm.20359](https://doi.org/10.1002/hbm.20359)
