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
- License is LGPL-2.1-only, as tedana (`LICENSE` is tedana's file byte for byte; the tedana,
  mapca and GIFT developers permitted it on 2026-10-18, see `NOTICE`). Keep `Cargo.toml`,
  `js/package.json`, `LICENSE` and `NOTICE` in step (npm's prepack copies `LICENSE` and `NOTICE`
  into js/; release archives include both).
- Open: the threaded build links wasm-bindgen-rayon (Apache-2.0 only, the one such crate). GNU
  counts Apache-2.0 compatible only with version-3 licences; whether LGPL-2.1 section 6 covers
  the combined wasm needs a legal determination. Until then do not distribute `dist/threaded`
  anywhere: npm ships `dist/single` only (release.yml enforces it), and the Neurodesk webapp
  (`webapps/apps/me-ica`) installs `@niivue/meica` from npm and stages only `dist/single`
  (`scripts/copy-runtime.mjs`).
- Open items are listed at the end of `DEVELOPMENT.md`.
