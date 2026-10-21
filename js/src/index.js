// Loader for meica's WebAssembly builds (scripts/build-wasm.mjs): dist/threaded needs a
// cross-origin isolated page; dist/single runs anywhere, Node included.

// Not literals, so bundlers neither resolve the package's dist/ nor pull node:fs into a browser bundle.
const DIST = "../dist/";
const NODE_FS = "node:fs/promises";

const bytes = (value) => (value instanceof Uint8Array ? value : new Uint8Array(value));

// One initialisation per module URL: import() reuses the module, and a second initThreadPool
// would fail installing rayon's global pool.
const loads = new Map();

/** True where the threaded build can run (SharedArrayBuffer and cross-origin isolation). */
export const canThread = () => globalThis.crossOriginIsolated === true && typeof SharedArrayBuffer !== "undefined";

/**
 * Load a build and return its module. `threads` > 1 picks the threaded build (when `canThread()`)
 * and starts that many workers. `baseUrl` is where the `threaded/` and `single/` folders are
 * served from (default: this package's `dist/`). Without a `threaded/` build there (the npm
 * package ships none for now), it falls back to `single/`. Later calls for
 * the same build share the first call's module and workers.
 */
export async function loadMeica({ threads = Math.min(8, globalThis.navigator?.hardwareConcurrency || 1), baseUrl } = {}) {
  const base = baseUrl ? new URL(baseUrl, globalThis.location?.href) : new URL(DIST, import.meta.url);
  const url = (build) => new URL(`${build}/meica_wasm.js`, base.href.endsWith("/") ? base : `${base}/`);
  if (threads > 1 && canThread()) return load(url("threaded"), threads).catch(() => load(url("single")));
  return load(url("single"));
}

function load(moduleUrl, threads) {
  if (!loads.has(moduleUrl.href)) {
    const loading = init(moduleUrl, threads);
    loading.catch(() => loads.delete(moduleUrl.href));
    loads.set(moduleUrl.href, loading);
  }
  return loads.get(moduleUrl.href);
}

/** Import and instantiate a build; start `threads` workers when given. */
async function init(moduleUrl, threads) {
  const api = await import(/* @vite-ignore */ moduleUrl.href);
  if (moduleUrl.protocol === "file:") {
    // Node: fetch() cannot read file URLs, so hand wasm-bindgen the bytes.
    const { readFile } = await import(/* @vite-ignore */ NODE_FS);
    await api.default({ module_or_path: await readFile(new URL("meica_wasm_bg.wasm", moduleUrl)) });
  } else {
    await api.default();
  }
  if (threads) await api.initThreadPool(threads);
  return api;
}

/**
 * Run tedana's workflow. `echoes` are `{ name, bytes }` in ascending TE order (NIfTI or NIfTI-gz);
 * `tes` are seconds or milliseconds; `options` uses tedana's flag names (`tree`, `tedpca`, `seed`,
 * `maxit`, `maxrestart`, `prefix`, `dummy_scans`, plus `external`: the regressors TSV's text);
 * `mask` (tedana --mask) is optional NIfTI bytes on the echoes' grid. Returns the summary and the
 * BIDS output files as `{ name, bytes }`.
 */
export function runMeica({ api, echoes, tes, options = {}, mask, onProgress = () => {} }) {
  const previous = globalThis.meicaProgress;
  globalThis.meicaProgress = onProgress;
  const session = new api.Session();
  try {
    for (const echo of echoes) session.add_echo(echo.name, bytes(echo.bytes));
    if (mask) session.set_mask(bytes(mask));
    const summary = JSON.parse(session.run(JSON.stringify({ ...options, tes })));
    const files = [];
    for (let i = 0; i < session.file_count(); i++) files.push({ name: session.file_name(i), bytes: session.take_file(i) });
    return { summary, files };
  } finally {
    globalThis.meicaProgress = previous;
    // After a Rust panic the session is still borrowed and free() throws, hiding the real error.
    try { session.free(); } catch {}
  }
}

/**
 * `meica --denoise`: apply a finished run's classification (`mixing`: desc-ICA_mixing.tsv text,
 * `metrics`: desc-tedana_metrics.tsv text) to another series of the same run, e.g. one resampled
 * to an anatomical space. Its first `dummyScans` volumes pass through. Returns float32 NIfTI bytes.
 */
export function denoiseMeica({ api, bold, mixing, metrics, dummyScans = 0, gzip = false }) {
  return api.denoise(bytes(bold), mixing, metrics, dummyScans, gzip);
}

/**
 * `meica --moco` in the browser: niimath motion-corrects the echoes (echo 1's BOLD reference,
 * dummy count and motion, applied to every echo in one interpolation) and writes fMRIPrep-named
 * confounds; brainchop-mindgrab masks the reference unless `mask` is given. `run(argv, { files })`
 * executes one command line over in-memory files (`has`, `bytes`, `put`): fastprep's tool
 * drivers (`niimath`, `brainchop-mindgrab`) fit as they are. TR comes from echo 1's header. Returns
 * `{ echoes, mask, confounds, options }` for `runMeica`; the caller's mask, `external` and
 * non-zero `dummy_scans` win, and the confounds become `external` for trees that take regressors.
 */
export async function prepareMoco({ api, echoes, run, mask, options = {} }) {
  const dir = "/moco";
  const store = new Map();
  const files = { has: (p) => store.has(p), bytes: async (p) => store.get(p), put: (p, b) => store.set(p, b) };
  const step = async (argv) => {
    const r = await run(argv, { files });
    if (r.code) throw new Error(`${argv[0]} failed: ${r.stderr || r.stdout}`);
  };
  const gz = (b) => b[0] === 0x1f && b[1] === 0x8b;
  const inputs = echoes.map((e, i) => {
    const b = bytes(e.bytes);
    const path = `${dir}/echo-${i + 1}.nii${gz(b) ? ".gz" : ""}`;
    files.put(path, b);
    return path;
  });
  const tr = api.nifti_tr(store.get(inputs[0]));
  const [nss, maskPath, confoundsPath] = api.moco_paths(dir);
  await step(api.moco_estimate(inputs[0], dir));
  const dummy_scans = options.dummy_scans || api.moco_dummy_scans(new TextDecoder().decode(store.get(nss)));
  if (mask) files.put(maskPath, bytes(mask));
  else await step(api.moco_mask(dir));
  const corrected = [];
  for (const [i, input] of inputs.entries()) {
    // niimath's browser driver writes NIFTI_GZ whatever the extension, so name it so.
    const out = `${dir}/moco-${i + 1}.nii.gz`;
    await step(api.moco_apply(input, dir, out, i === 0, tr, dummy_scans));
    corrected.push({ name: echoes[i].name, bytes: store.get(out) });
  }
  const confounds = new TextDecoder().decode(store.get(confoundsPath));
  const tree = options.tree ?? "tedana_orig";
  const external = options.external ?? (api.tree_takes_regressors(tree) ? confounds : undefined);
  return { echoes: corrected, mask: store.get(maskPath), confounds, options: { ...options, dummy_scans, ...(external && { external }) } };
}
