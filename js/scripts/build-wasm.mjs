// Both WebAssembly builds into dist/: threaded (wasm-bindgen-rayon; nightly build-std with
// atomics, 4 GiB maximum memory) and single-threaded (stable toolchain).
import { execFileSync } from "node:child_process";
import { globSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";

const crate = fileURLToPath(new URL("../../wasm", import.meta.url));
const dist = (name) => fileURLToPath(new URL(`../dist/${name}`, import.meta.url));
const NIGHTLY = "nightly-2025-11-15";
// Panic locations would otherwise publish the builder's home directory in the wasm.
const remap = `--remap-path-prefix=${homedir()}=~`;
const atomics = [
  remap,
  "-C target-feature=+atomics,+bulk-memory",
  "-C link-arg=--shared-memory",
  "-C link-arg=--max-memory=4294967296",
  "-C link-arg=--import-memory",
  "-C link-arg=--export=__wasm_init_tls",
  "-C link-arg=--export=__tls_size",
  "-C link-arg=--export=__tls_align",
  "-C link-arg=--export=__tls_base",
].join(" ");
const pack = (toolchain, out, cargoArgs, env = {}) =>
  execFileSync("rustup", ["run", toolchain, "wasm-pack", "build", crate, "--target", "web", "--release", "--no-pack",
    "--out-dir", dist(out), "--out-name", "meica_wasm", "--", ...cargoArgs], { stdio: "inherit", env: { ...process.env, ...env } });

rmSync(dist(""), { recursive: true, force: true });
pack(NIGHTLY, "threaded", ["-Z", "build-std=panic_abort,std"], { CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS: atomics });
pack("stable", "single", ["--no-default-features"], { CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS: remap });
for (const file of globSync(`${dist("")}/**/{.gitignore,*.js}`, { dot: true })) {
  if (file.endsWith(".gitignore")) rmSync(file);
  else writeFileSync(file, readFileSync(file, "utf8").replaceAll("\r\n", "\n"));
}
