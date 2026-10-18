#!/usr/bin/env node
// Run the example through a single-threaded WebAssembly build and require every output file to
// be byte-identical to the native CLI's.
//
// Usage: node validation/wasm_parity.mjs WASM_PKG_DIR EXAMPLE_DIR NATIVE_OUT_DIR [TREE REGRESSORS_TSV]
// (WASM_PKG_DIR is js/dist/single from `make wasm`; see Makefile).
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, realpathSync } from "node:fs";
import { join, relative } from "node:path";
import { pathToFileURL } from "node:url";

// Canonical, as the CLI writes its registry's input paths.
const [pkg, example, native] = process.argv.slice(2, 5).map((p) => realpathSync(p));
const [tree, regressors] = process.argv.slice(5);
const extra = tree ? { tree, external: readFileSync(regressors, "utf8") } : {};
const api = await import(pathToFileURL(join(pkg, "meica_wasm.js")).href);
await api.default({ module_or_path: readFileSync(join(pkg, "meica_wasm_bg.wasm")) });
globalThis.meicaProgress = () => {};

// Same input names and command line as the native run, so its JSON files can match too.
const nativeCommand = JSON.parse(readFileSync(join(native, "dataset_description.json"), "utf8")).GeneratedBy[0].Command;
const session = new api.Session();
for (const e of [1, 2, 3]) {
  const file = join(example, `sub-01_task-motor_echo-${e}_bold.nii.gz`);
  session.add_echo(relative(native, file), readFileSync(file));
}
const started = performance.now();
const summary = JSON.parse(session.run(JSON.stringify({ tes: [15.2, 35.32, 55.42], command: nativeCommand, ...extra })));
const seconds = ((performance.now() - started) / 1000).toFixed(1);

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const produced = new Map();
for (let i = 0; i < session.file_count(); i++) produced.set(session.file_name(i), session.take_file(i));
const expected = readdirSync(native).sort();
let failures = 0;
for (const name of expected) {
  const got = produced.get(name);
  const same = got && sha(got) === sha(readFileSync(join(native, name)));
  if (!same) {
    failures++;
    console.log(`DIFFERS ${name}${got ? "" : " (missing)"}`);
  }
}
for (const name of produced.keys()) {
  if (!expected.includes(name)) {
    failures++;
    console.log(`EXTRA ${name}`);
  }
}
console.log(`WebAssembly (${seconds} s): ${summary.accepted} accepted, ${summary.rejected} rejected; ` +
  (failures ? `${failures} of ${expected.length} files differ from native` : `all ${expected.length} files byte-identical to native`));
process.exit(failures ? 1 : 0);
