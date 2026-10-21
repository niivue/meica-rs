import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, mkdtempSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { denoiseMeica, loadMeica, prepareMoco, runMeica } from "../src/index.js";

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

test("prepareMoco runs niimath and mindgrab through the caller's driver and feeds runMeica", { skip: !existsSync(new URL("../dist/single/meica_wasm.js", import.meta.url)) && "run npm run build first" }, async () => {
  const api = await loadMeica({ threads: 1 });
  const lines = [];
  // A fake driver: records each command line and writes the files the real tools would.
  const run = async (argv, { files }) => {
    lines.push(argv);
    const out = (flag) => argv[argv.indexOf(flag) + 1];
    if (argv.includes("-boldref")) files.put(argv[argv.indexOf("-boldref") + 2], new TextEncoder().encode('{"dummy_scans": 2}'));
    if (argv[0] === "brainchop-mindgrab") files.put(out("--mask"), Uint8Array.of(9));
    if (argv.includes("-confounds")) files.put(out("-confounds"), new TextEncoder().encode("trans_x\n0\n"));
    if (argv.includes("-resample")) files.put(argv.at(-1), Uint8Array.of(argv.length));
    return { code: 0, stdout: "", stderr: "" };
  };
  // Header-only NIfTI-1 echoes: prepareMoco reads TR (pixdim[4]) from echo 1.
  const header = () => {
    const v = new DataView(new ArrayBuffer(352));
    v.setInt32(0, 348, true);
    [4, 2, 2, 2, 3].forEach((d, i) => v.setInt16(40 + 2 * i, d, true));
    v.setInt16(70, 16, true);
    v.setInt16(72, 32, true);
    [1, 1, 1, 1, 2.5].forEach((p, i) => v.setFloat32(76 + 4 * i, p, true));
    v.setFloat32(108, 352, true);
    new Uint8Array(v.buffer).set([0x6e, 0x2b, 0x31], 344);
    return new Uint8Array(v.buffer);
  };
  const echoes = [1, 2, 3].map((e) => ({ name: `e${e}.nii.gz`, bytes: header() }));
  const r = await prepareMoco({ api, echoes, run, options: { tree: "motion_f" } });
  assert.deepEqual(lines.map((a) => a[0]), ["niimath", "brainchop-mindgrab", "niimath", "niimath", "niimath"]);
  assert.equal(lines[0][1], "/moco/echo-1.nii");
  assert.equal(lines[2][lines[2].indexOf("-tr") + 1], "2.5");
  assert.equal(lines[2][lines[2].indexOf("-dummy") + 1], "2");
  assert.ok(lines.every((a) => !a.includes("-")), "no stdout output in the browser");
  assert.deepEqual(r.echoes.map((e) => e.name), ["e1.nii.gz", "e2.nii.gz", "e3.nii.gz"]);
  assert.deepEqual(r.mask, Uint8Array.of(9));
  assert.equal(r.options.dummy_scans, 2);
  assert.equal(r.options.external, "trans_x\n0\n");
  // A given mask skips mindgrab; the default tree takes no regressors.
  lines.length = 0;
  const own = await prepareMoco({ api, echoes, run, mask: Uint8Array.of(7), options: { dummy_scans: 4 } });
  assert.ok(!lines.some((a) => a[0] === "brainchop-mindgrab"));
  assert.equal(lines[1][lines[1].indexOf("-dummy") + 1], "4");
  assert.equal(own.options.external, undefined);
});

test("denoiseMeica removes rejected components and passes dummies through", { skip: !existsSync(new URL("../dist/single/meica_wasm.js", import.meta.url)) && "run npm run build first" }, async () => {
  // 2 voxels x 5 volumes (float32), 1 dummy; one mixing column, rejected.
  const series = [[9, 1, 2, 3, 4], [9, 4, 6, 6, 4]];
  const nifti = new DataView(new ArrayBuffer(352 + 40));
  nifti.setInt32(0, 348, true);
  [4, 2, 1, 1, 5, 1, 1, 1].forEach((d, i) => nifti.setInt16(40 + 2 * i, d, true));
  nifti.setInt16(70, 16, true);
  nifti.setInt16(72, 32, true);
  nifti.setFloat32(108, 352, true);
  series.forEach((ts, v) => ts.forEach((x, t) => nifti.setFloat32(352 + 4 * (t * 2 + v), x, true)));
  const api = await loadMeica({ threads: 1 });
  const metrics = "Component\tclassification\nICA_00\t";
  const run = (label) => new DataView(denoiseMeica({ api, bold: new Uint8Array(nifti.buffer), mixing: "ICA_00\n-1.5\n-0.5\n0.5\n1.5\n", metrics: metrics + label, dummyScans: 1 }).buffer);
  const value = (out, v, t) => out.getFloat32(352 + 4 * (t * 2 + v), true);
  const rejected = run("rejected");
  assert.equal(value(rejected, 0, 0), 9);
  // voxel 0 is a pure ramp: only its mean (2.5) remains; voxel 1 has no ramp: unchanged.
  for (let t = 1; t < 5; t++) assert.ok(Math.abs(value(rejected, 0, t) - 2.5) < 1e-6);
  for (let t = 1; t < 5; t++) assert.equal(value(rejected, 1, t), series[1][t]);
  const accepted = run("accepted");
  for (let t = 0; t < 5; t++) assert.equal(value(accepted, 0, t), series[0][t]);
  assert.throws(() => denoiseMeica({ api, bold: new Uint8Array(nifti.buffer), mixing: "ICA_00\n1\n", metrics: metrics + "rejected" }));
});
