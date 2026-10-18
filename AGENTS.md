# meica-rs

Rust port of tedana's ME-ICA (core, cli `meica`, wasm) plus the npm package `@niivue/meica` (js/).
`DEVELOPMENT.md` is the maintainer contract: numerics, references and traps.

- Equivalence gates only tighten. Run them as single make invocations (they share builds):
  `make reference MEICA_HF_DIR=DIR`, then
  `make test-real check-outputs check-external wasm-parity lint PYTHON=python MEICA_REFERENCE_DIR=DIR/me-ica/motor-3echo-moco`
  (and `test-real check-outputs` with `motor-3echo`).
- Reference data lives on Hugging Face (`neurodeskorg/webapps`, `me-ica/…`, pinned by
  `MEICA_HF_REV` in the Makefile). Re-pin after uploading regenerated references.
- `core/trees/motion_*.json` must stay byte-identical to the HF `trees/` copies, and change only
  together with their tedana references. Other trees are tedana's, verbatim.
- Output bytes are pinned downstream (Neurodesk webapps `apps/me-ica/e2e/expected.json`); a change
  to any output needs those hashes re-recorded.
- License is GPL-2.0-only pending LGPL confirmation (README "License"); keep `Cargo.toml`,
  `js/package.json` and `LICENSE` in step when it changes (npm's prepack copies `LICENSE` and
  `NOTICE` into js/). No Apache-2.0 code: it cannot join a GPL-2.0-only work.
- Open: wasm-bindgen-rayon (threaded build, `threads` feature) is Apache-2.0 only, the one such
  crate. Fix by relicensing to (L)GPL-3, shipping single-threaded only, or a clean-room worker
  pool. Undecided: npm ships `dist/single` only until resolved. The Neurodesk webapp
  (`webapps/apps/me-ica`, links `meica-rs/js`) still deploys the threaded build
  (`scripts/copy-runtime.mjs` copies all of `dist/`): copy `dist/single` only, or resolve the
  licence, before deploying.
- Open items are listed at the end of `DEVELOPMENT.md`.
