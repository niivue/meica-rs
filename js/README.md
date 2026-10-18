# @niivue/meica

Multi-echo ICA denoising (a Rust port of [tedana](https://github.com/ME-ICA/tedana)) as
WebAssembly, from [meica-rs](https://github.com/niivue/meica-rs). Outputs match the native `meica`
CLI byte for byte.

```js
import { loadMeica, runMeica } from "@niivue/meica";
const api = await loadMeica();
const { summary, files } = runMeica({ api, echoes, tes: [15.2, 35.32, 55.42], options: { tree: "tedana_orig" } });
```

- `loadMeica({ threads, baseUrl })` loads `single/` (browser or Node 22+). The package ships only
  `dist/single`; when the page is cross-origin isolated and `threads` > 1 (default: up to 8
  cores), it first tries `threaded/`, which only a self-built `baseUrl` serves, and falls back to
  `single/`. `baseUrl` points at a copy of `dist/` when a bundler does not serve the package's
  files.
- `runMeica({ api, echoes, tes, options, onProgress })`: `echoes` are `{ name, bytes }` NIfTI
  files in echo order; `options` use tedana's flag names (`tree`, `tedpca`, `seed`, `maxit`,
  `maxrestart`, `prefix`, and `external`, the regressors TSV's text). Returns `{ summary, files }`.
  Run it in a Web Worker: it blocks for seconds.

Memory may grow to 4 GiB; measured peaks are in the repository's `DEVELOPMENT.md`.
GPL-2.0-only; see `NOTICE` and the repository README's License section.

Build from a checkout: `npm run build` (see the repository README for toolchains).
