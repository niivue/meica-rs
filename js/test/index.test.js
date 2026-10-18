import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, mkdtempSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { loadMeica, runMeica } from "../src/index.js";

/** A stand-in for the WebAssembly Session that records what the wrapper passes. */
function fakeApi(log) {
  return {
    Session: class {
      add_echo(name, data) {
        log.push(["echo", name, data.length]);
      }
      run(options) {
        log.push(["run", JSON.parse(options)]);
        globalThis.meicaProgress("halfway");
        return JSON.stringify({ n_components: 2, accepted: 1, rejected: 1, seed: 42, n_iter: 9 });
      }
      file_count() {
        return 1;
      }
      file_name() {
        return "desc-denoised_bold.nii.gz";
      }
      take_file() {
        return Uint8Array.of(1, 2, 3);
      }
      free() {
        log.push(["free"]);
      }
    },
  };
}

test("runMeica passes echoes and options, reports progress and returns files", () => {
  const log = [];
  const progress = [];
  const result = runMeica({
    api: fakeApi(log),
    echoes: [{ name: "e1.nii.gz", bytes: [1, 2] }, { name: "e2.nii.gz", bytes: Uint8Array.of(3) }],
    tes: [15, 40],
    options: { tree: "minimal" },
    onProgress: (m) => progress.push(m),
  });
  assert.deepEqual(log, [["echo", "e1.nii.gz", 2], ["echo", "e2.nii.gz", 1], ["run", { tree: "minimal", tes: [15, 40] }], ["free"]]);
  assert.deepEqual(progress, ["halfway"]);
  assert.equal(result.summary.accepted, 1);
  assert.deepEqual(result.files, [{ name: "desc-denoised_bold.nii.gz", bytes: Uint8Array.of(1, 2, 3) }]);
  assert.equal(globalThis.meicaProgress, undefined);
});

test("loadMeica falls back to the single-threaded build in Node", { skip: !existsSync(new URL("../dist/single/meica_wasm.js", import.meta.url)) && "run npm run build first" }, async () => {
  const [api, again] = await Promise.all([loadMeica({ threads: 4 }), loadMeica()]);
  assert.equal(api, again);
  assert.equal(api.initThreadPool, undefined);
  assert.throws(() => runMeica({ api, echoes: [], tes: [15, 30, 45] }), /echo/);
});

test("loadMeica falls back to single when no threaded build is served", { skip: !existsSync(new URL("../dist/single/meica_wasm.js", import.meta.url)) && "run npm run build first" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "meica-"));
  symlinkSync(fileURLToPath(new URL("../dist/single", import.meta.url)), join(dir, "single"));
  globalThis.crossOriginIsolated = true;
  try {
    const api = await loadMeica({ threads: 4, baseUrl: pathToFileURL(dir) });
    assert.equal(api.initThreadPool, undefined);
  } finally {
    delete globalThis.crossOriginIsolated;
    rmSync(dir, { recursive: true });
  }
});
