TMPDIR ?= /tmp
PYTHON ?= python3

.PHONY: reference build test test-real check-outputs check-external wasm wasm-parity lint fixtures

# Example, tedana references and stage files for both examples from Hugging Face, pinned. Then use
# MEICA_REFERENCE_DIR=$(MEICA_HF_DIR)/me-ica/motor-3echo (or motor-3echo-moco, which adds the
# external-regressor trees and references).
MEICA_HF_REV = 121b25dde5caffc8f880eec87cca29b4e569b7a9
MEICA_HF_DIR ?= $(TMPDIR)/meica-hf
reference:
	hf download neurodeskorg/webapps --repo-type dataset --revision $(MEICA_HF_REV) \
		--include "me-ica/*" --local-dir $(MEICA_HF_DIR)

build:
	cargo build --release --workspace

test:
	cargo test --workspace --release

# MEICA_REFERENCE_DIR holds example/, reference/ and stages/ (DEVELOPMENT.md "Reference data").
test-real:
	MEICA_REQUIRE_REFERENCE=1 cargo test --workspace --release

run_example = ./target/release/meica -d $(foreach e,1 2 3,$(MEICA_REFERENCE_DIR)/example/sub-01_task-motor_echo-$(e)_bold.nii.gz) \
	-e 15.2 35.32 55.42 --out-dir $(1) --quiet

# Run the CLI on the example and compare every output with tedana's (needs nibabel, numpy).
check-outputs: build
	$(PYTHON) validation/test_compare_outputs.py
	rm -rf $(TMPDIR)/meica-check
	$(call run_example,$(TMPDIR)/meica-check)
	$(PYTHON) validation/compare_outputs.py $(MEICA_REFERENCE_DIR)/reference $(TMPDIR)/meica-check

# External-regressor trees against tedana's outputs in external/<tree>/: tedana's demo trees on
# external/regressors.tsv (gen_external_regressors.py), and the bundled fMRIPrep-named motion
# trees (trees/<tree>.json, which must equal core/trees/) on the example's confounds.
DEMO_TREES = demo_external_regressors_single_model demo_external_regressors_motion_task_models demo_external_aroma
MOTION_TREES = motion_f motion_aroma
CONFOUNDS = $(MEICA_REFERENCE_DIR)/example/sub-01_task-motor_desc-confounds_timeseries.tsv
external_args = $(if $(filter $(1),$(DEMO_TREES)),--tree $(1) --external $(MEICA_REFERENCE_DIR)/external/regressors.tsv,--tree $(MEICA_REFERENCE_DIR)/trees/$(1).json --external $(CONFOUNDS))
check-external: build
	set -e; for t in $(MOTION_TREES); do cmp core/trees/$$t.json $(MEICA_REFERENCE_DIR)/trees/$$t.json; done
	set -e; $(foreach t,$(DEMO_TREES) $(MOTION_TREES), \
		rm -rf $(TMPDIR)/meica-$(t); \
		$(call run_example,$(TMPDIR)/meica-$(t)) $(call external_args,$(t)); \
		$(PYTHON) validation/compare_outputs.py $(MEICA_REFERENCE_DIR)/external/$(t) $(TMPDIR)/meica-$(t);)

# Both WebAssembly builds into js/dist (threaded: nightly-2025-11-15 build-std; single: stable).
wasm:
	cd js && node scripts/build-wasm.mjs

# The single-threaded build in Node must write every output byte-identical to the native CLI's.
wasm-parity: build wasm
	rm -rf $(TMPDIR)/meica-native
	$(call run_example,$(TMPDIR)/meica-native)
	node validation/wasm_parity.mjs js/dist/single $(MEICA_REFERENCE_DIR)/example $(TMPDIR)/meica-native
	set -e; $(foreach t,$(DEMO_TREES) $(MOTION_TREES), \
		rm -rf $(TMPDIR)/meica-native; \
		$(call run_example,$(TMPDIR)/meica-native) $(call external_args,$(t)); \
		node validation/wasm_parity.mjs js/dist/single $(MEICA_REFERENCE_DIR)/example $(TMPDIR)/meica-native $(t) \
			$(if $(filter $(t),$(DEMO_TREES)),$(MEICA_REFERENCE_DIR)/external/regressors.tsv,$(CONFOUNDS));)

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy -p meica-wasm --target wasm32-unknown-unknown --no-default-features -- -D warnings
	cargo fmt --all --check

# Needs tedana's reference environment (numpy, scipy, mpmath); see DEVELOPMENT.md.

fixtures:
	$(PYTHON) validation/gen_fixtures.py core/tests/fixtures
