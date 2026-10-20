# @niivue/meica

Multi-echo ICA denoising (a Rust port of [tedana](https://github.com/ME-ICA/tedana)) as
WebAssembly, from [meica-rs](https://github.com/niivue/meica-rs). `runMeica` outputs match the native
`meica` CLI byte for byte (`prepareMoco` results depend on the niimath and mindgrab builds).

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
- `runMeica({ api, echoes, tes, options, mask, onProgress })`: `echoes` are `{ name, bytes }` NIfTI
  files in echo order; `options` use tedana's flag names (`tree`, `tedpca`, `seed`, `maxit`,
  `maxrestart`, `prefix`, `dummy_scans`, and `external`, the regressors TSV's text); `mask` is
  optional NIfTI bytes (tedana `--mask`). Returns `{ summary, files }`.
  Run it in a Web Worker: it blocks for seconds.
- `prepareMoco({ api, echoes, run, mask, options })` is `meica --moco` in the browser: `run(argv,
  { files })` runs one niimath or brainchop-mindgrab command line over in-memory files, as
  fastprep's tool drivers do. TR comes from echo 1's header, as in the CLI. Returns `{ echoes, mask, confounds, options }` for `runMeica`.

Memory may grow to 4 GiB; measured peaks are in the repository's `DEVELOPMENT.md`.
LGPL-2.1-only, as tedana; see `LICENSE` and `NOTICE` (attributions, including the msun notice).

Build from a checkout: `npm run build` (see the repository README for toolchains).
