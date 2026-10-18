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
 * package ships none until its licence is settled), it falls back to `single/`. Later calls for
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
 * `maxit`, `maxrestart`, `prefix`, plus `external`: the regressors TSV's text). Returns the
 * summary and the BIDS output files as `{ name, bytes }`.
 */
export function runMeica({ api, echoes, tes, options = {}, onProgress = () => {} }) {
  const previous = globalThis.meicaProgress;
  globalThis.meicaProgress = onProgress;
  const session = new api.Session();
  try {
    for (const echo of echoes) session.add_echo(echo.name, bytes(echo.bytes));
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
