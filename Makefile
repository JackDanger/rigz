# gzippy - The Fastest Parallel Gzip
# Build and test infrastructure
#
# Quick tests (<30s) run with 'make' or 'make quick' - for AI tools and iteration
# Full perf tests (10+ min) run with 'make perf-full' - for humans at release time

# Build configuration - submodules are in ./gzip, ./pigz, ./isa-l, ./zopfli, and ./rapidgzip
GZIP_DIR := ./gzip
PIGZ_DIR := ./pigz
ISAL_DIR := ./isa-l
ZOPFLI_DIR := ./zopfli
RAPIDGZIP_DIR := ./rapidgzip
GZIPPY_DIR := .
TEST_DATA_DIR := test_data
RESULTS_DIR := test_results

# Build targets
GZIPPY_BIN := $(GZIPPY_DIR)/target/release/gzippy
UNGZIPPY_BIN := $(GZIPPY_DIR)/target/release/ungzippy
PIGZ_BIN := $(PIGZ_DIR)/pigz
IGZIP_BIN := $(ISAL_DIR)/build/igzip
ZOPFLI_BIN := $(ZOPFLI_DIR)/zopfli
RAPIDGZIP_BIN := $(RAPIDGZIP_DIR)/librapidarchive/build/src/tools/rapidgzip

# Prefer local gzip build, fall back to system gzip
GZIP_BIN := $(shell if [ -x $(GZIP_DIR)/gzip ]; then echo $(GZIP_DIR)/gzip; else echo $$(which gzip); fi)
SYSTEM_GZIP := $(shell which gzip)

.PHONY: all build quick perf-full test-data test-data-quick clean help validate deps

# =============================================================================
# Default target: quick benchmark for fast iteration (< 30 seconds)
# =============================================================================
all: quick

# =============================================================================
# Build targets
# =============================================================================

build: $(GZIPPY_BIN) $(UNGZIPPY_BIN)

deps: $(PIGZ_BIN) $(IGZIP_BIN) $(ZOPFLI_BIN) $(RAPIDGZIP_BIN)
	@# Try to build gzip, but don't fail if it doesn't work
	@$(MAKE) $(GZIP_DIR)/gzip 2>/dev/null || true
	@echo "✓ Dependencies ready (gzip, pigz, igzip, zopfli, rapidgzip)"

$(GZIP_DIR)/gzip:
	@echo "Building gzip from source..."
	@# Fix autotools timestamps to prevent regeneration
	@cd $(GZIP_DIR) && find . -name "*.in" -exec touch {} \; 2>/dev/null; \
		touch configure aclocal.m4 Makefile.in 2>/dev/null || true
	@cd $(GZIP_DIR) && ./configure --quiet 2>/dev/null || true
	@if $(MAKE) -C $(GZIP_DIR) -j4 2>/dev/null; then \
		echo "✓ Built gzip from source"; \
	else \
		echo "⚠ gzip build failed, using system gzip: $(SYSTEM_GZIP)"; \
	fi

$(PIGZ_BIN):
	@echo "Building pigz from source..."
	@$(MAKE) -C $(PIGZ_DIR) pigz 2>&1 || (echo "  Cleaning and rebuilding..." && $(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 && $(MAKE) -C $(PIGZ_DIR) pigz)
	@echo "✓ Built pigz"

$(IGZIP_BIN):
	@echo "Building igzip (ISA-L) from source..."
	@mkdir -p $(ISAL_DIR)/build
	@cd $(ISAL_DIR)/build && cmake .. >/dev/null 2>&1 && make -j4 igzip 2>&1 | grep -E "(Built|error)" || true
	@echo "✓ Built igzip"

$(ZOPFLI_BIN):
	@echo "Building zopfli from source..."
	@$(MAKE) -C $(ZOPFLI_DIR) zopfli 2>&1 | grep -E "(cc|g\+\+|error)" || true
	@echo "✓ Built zopfli"

$(RAPIDGZIP_BIN):
	@echo "Building rapidgzip from source..."
	@cd $(RAPIDGZIP_DIR) && git submodule update --init --recursive >/dev/null 2>&1 || true
	@mkdir -p $(RAPIDGZIP_DIR)/librapidarchive/build
	@cd $(RAPIDGZIP_DIR)/librapidarchive/build && cmake .. >/dev/null 2>&1 && make -j4 rapidgzip 2>&1 | grep -E "(Built|Linking|error)" || true
	@echo "✓ Built rapidgzip"

$(GZIPPY_BIN): FORCE
	@echo "Building gzippy..."
	@cd $(GZIPPY_DIR) && cargo build --release 2>&1 | grep -E "(Compiling gzippy|Finished|error)" || true
	@echo "✓ Built gzippy"

# Create ungzippy symlink (like unpigz)
$(UNGZIPPY_BIN): $(GZIPPY_BIN)
	@ln -sf gzippy $(UNGZIPPY_BIN)
	@echo "✓ Created ungzippy symlink"

FORCE:

# =============================================================================
# Quick benchmark (~30 seconds) - for AI tools and fast iteration
# =============================================================================
quick: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@python3 scripts/perf.py --sizes 1,10 --levels 6 --threads 1,4

# =============================================================================
# Full performance tests (10+ minutes) - for humans at release time
# =============================================================================
perf-full: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@mkdir -p $(RESULTS_DIR)
	@python3 scripts/perf.py --full 2>&1 | tee $(RESULTS_DIR)/perf_full_$$(date +%Y%m%d_%H%M%S).log

# Generate test data files using Python script
# Uses test_data/text-1MB.txt (Proust) as seed for highly-compressible text
test-data:
	@python3 scripts/generate_test_data.py --output-dir $(TEST_DATA_DIR) --size 10
	@python3 scripts/generate_test_data.py --output-dir $(TEST_DATA_DIR) --size 100

# Generate just 10MB test files (faster for quick testing)
test-data-quick:
	@python3 scripts/generate_test_data.py --output-dir $(TEST_DATA_DIR) --size 10

# =============================================================================
# Validation target - cross-tool compression/decompression matrix
# =============================================================================
validate: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@python3 scripts/validate.py

# Validation with JSON output (run tests, save results)
validate-json: $(GZIPPY_BIN) $(UNGZIPPY_BIN) $(PIGZ_BIN) deps
	@mkdir -p $(RESULTS_DIR)
	@python3 scripts/validate.py --json -o $(RESULTS_DIR)/validation.json
	@echo "✓ Results saved to $(RESULTS_DIR)/validation.json"

# Run validation + generate charts (full workflow)
validation-chart: validate-json render-chart

# Render charts from existing JSON (fast iteration on chart rendering)
render-chart:
	@if [ ! -f $(RESULTS_DIR)/validation.json ]; then \
		echo "Error: $(RESULTS_DIR)/validation.json not found. Run 'make validate-json' first."; \
		exit 1; \
	fi
	@echo ""
	@python3 scripts/validation_chart.py $(RESULTS_DIR)/validation.json
	@python3 scripts/validation_chart.py $(RESULTS_DIR)/validation.json --html > $(RESULTS_DIR)/validation.html
	@echo ""
	@echo "✓ HTML chart: $(RESULTS_DIR)/validation.html"

# =============================================================================
# Lint target
# =============================================================================
lint:
	@echo "Running rustfmt..."
	@cargo fmt --all
	@echo "Running clippy..."
	@cargo clippy --release -- -D warnings
	@echo "✓ Lint passed"

lint-check:
	@echo "Checking formatting..."
	@cargo fmt --all --check
	@cargo clippy --release -- -D warnings
	@echo "✓ Lint check passed"

# =============================================================================
# Install target
# =============================================================================
install: $(GZIPPY_BIN) $(UNGZIPPY_BIN)
	@echo "Installing to /usr/local/bin..."
	@install -m 755 $(GZIPPY_BIN) /usr/local/bin/gzippy
	@ln -sf gzippy /usr/local/bin/ungzippy
	@echo "✓ Installed gzippy and ungzippy"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@echo "Cleaning..."
	@rm -rf $(TEST_DATA_DIR) $(RESULTS_DIR) $(BENCH_RESULTS_DIR) $(BENCH_BIN_DIR)
	@$(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 || true
	@$(MAKE) -C $(GZIP_DIR) clean >/dev/null 2>&1 || true
	@cd $(GZIPPY_DIR) && cargo clean >/dev/null 2>&1
	@echo "✓ Cleaned"

# =============================================================================
# Benchmark Data Preparation (matches CI)
# =============================================================================

BENCHMARK_DIR := ./benchmark_data
BENCH_RESULTS_DIR := ./benchmark_results
BENCH_BIN_DIR := ./bench_bin
THREADS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Setup bin directory for multi-tool benchmarks
bench-bin: $(GZIPPY_BIN) $(PIGZ_BIN) $(IGZIP_BIN)
	@mkdir -p $(BENCH_BIN_DIR)
	@cp -f $(GZIPPY_BIN) $(BENCH_BIN_DIR)/
	@cp -f $(PIGZ_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(PIGZ_DIR)/unpigz $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(IGZIP_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(RAPIDGZIP_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@cp -f $(ZOPFLI_BIN) $(BENCH_BIN_DIR)/ 2>/dev/null || true
	@echo "✓ Benchmark binaries ready in $(BENCH_BIN_DIR)/"

# Benchmark data files
SILESIA_TAR := $(BENCHMARK_DIR)/silesia.tar
SILESIA_GZ := $(BENCHMARK_DIR)/silesia-gzip.tar.gz
SOFTWARE := $(BENCHMARK_DIR)/software.archive
SOFTWARE_GZ := $(BENCHMARK_DIR)/software.archive.gz
LOGS := $(BENCHMARK_DIR)/logs.txt
LOGS_GZ := $(BENCHMARK_DIR)/logs.txt.gz

.PHONY: bench-data bench-bin bench-decompress bench-decompress-all bench-compress bench-compress-all
.PHONY: bench-decompress-silesia bench-decompress-silesia-all
.PHONY: bench-decompress-software bench-decompress-software-all
.PHONY: bench-decompress-logs bench-decompress-logs-all
.PHONY: bench-compress-silesia-l1 bench-compress-silesia-l1-all
.PHONY: bench-compress-silesia-l6 bench-compress-silesia-l6-all
.PHONY: bench-compress-silesia-l9 bench-compress-silesia-l9-all
.PHONY: bench-compress-software-l1 bench-compress-software-l1-all
.PHONY: bench-compress-software-l6 bench-compress-software-l6-all
.PHONY: bench-compress-software-l9 bench-compress-software-l9-all
.PHONY: bench-compress-logs-l1 bench-compress-logs-l1-all
.PHONY: bench-compress-logs-l6 bench-compress-logs-l6-all
.PHONY: bench-compress-logs-l9 bench-compress-logs-l9-all
.PHONY: bench bench-all bench-exhaustive

bench-data:
	@chmod +x scripts/prepare_benchmark_data.sh
	./scripts/prepare_benchmark_data.sh all

# =============================================================================
# DECOMPRESSION BENCHMARKS
# =============================================================================

# --- Silesia (mixed binary/text) ---
bench-decompress-silesia: $(GZIPPY_BIN) bench-data
	@echo "=== Decompression: silesia (gzippy only) ==="
	RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_silesia -- --nocapture

bench-decompress-silesia-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_decompression.py \
		--binaries $(BENCH_BIN_DIR) --compressed-file $(SILESIA_GZ) --original-file $(SILESIA_TAR) \
		--threads $(THREADS) --archive-type silesia \
		--output $(BENCH_RESULTS_DIR)/decompress-silesia.json

# --- Software (source code patterns) ---
bench-decompress-software: $(GZIPPY_BIN) bench-data
	@echo "=== Decompression: software (gzippy only) ==="
	RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_software -- --nocapture

bench-decompress-software-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_decompression.py \
		--binaries $(BENCH_BIN_DIR) --compressed-file $(SOFTWARE_GZ) --original-file $(SOFTWARE) \
		--threads $(THREADS) --archive-type software \
		--output $(BENCH_RESULTS_DIR)/decompress-software.json

# --- Logs (repetitive data) ---
bench-decompress-logs: $(GZIPPY_BIN) bench-data
	@echo "=== Decompression: logs (gzippy only) ==="
	RUSTFLAGS="-C target-cpu=native" cargo test --release bench_cf_logs -- --nocapture

bench-decompress-logs-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_decompression.py \
		--binaries $(BENCH_BIN_DIR) --compressed-file $(LOGS_GZ) --original-file $(LOGS) \
		--threads $(THREADS) --archive-type logs \
		--output $(BENCH_RESULTS_DIR)/decompress-logs.json

# --- Combined decompression ---
bench-decompress: bench-decompress-silesia bench-decompress-software bench-decompress-logs

bench-decompress-all: bench-decompress-silesia-all bench-decompress-software-all bench-decompress-logs-all
	@echo ""
	@echo "=== Decompression Results ==="
	@for f in $(BENCH_RESULTS_DIR)/decompress-*.json; do \
		[ -f "$$f" ] && echo "--- $$(basename $$f .json) ---" && \
		python3 -c "import json,sys; d=json.load(open('$$f')); [print(f'  {r[\"tool\"]}: {r.get(\"speed_mbps\",0):.1f} MB/s') for r in d.get('results',[]) if 'error' not in r]" 2>/dev/null || true; \
	done

# =============================================================================
# COMPRESSION BENCHMARKS
# =============================================================================

# --- Silesia L1 (fast) ---
bench-compress-silesia-l1: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: silesia L1 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -1 -p$(THREADS) -c $(SILESIA_TAR) > /dev/null'

bench-compress-silesia-l1-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SILESIA_TAR) \
		--level 1 --threads $(THREADS) --content-type silesia \
		--output $(BENCH_RESULTS_DIR)/compress-silesia-l1.json

# --- Silesia L6 (default) ---
bench-compress-silesia-l6: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: silesia L6 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -6 -p$(THREADS) -c $(SILESIA_TAR) > /dev/null'

bench-compress-silesia-l6-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SILESIA_TAR) \
		--level 6 --threads $(THREADS) --content-type silesia \
		--output $(BENCH_RESULTS_DIR)/compress-silesia-l6.json

# --- Silesia L9 (best) ---
bench-compress-silesia-l9: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: silesia L9 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -9 -p$(THREADS) -c $(SILESIA_TAR) > /dev/null'

bench-compress-silesia-l9-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SILESIA_TAR) \
		--level 9 --threads $(THREADS) --content-type silesia \
		--output $(BENCH_RESULTS_DIR)/compress-silesia-l9.json

# --- Software L1/L6/L9 ---
bench-compress-software-l1: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: software L1 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -1 -p$(THREADS) -c $(SOFTWARE) > /dev/null'

bench-compress-software-l1-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SOFTWARE) \
		--level 1 --threads $(THREADS) --content-type software \
		--output $(BENCH_RESULTS_DIR)/compress-software-l1.json

bench-compress-software-l6: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: software L6 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -6 -p$(THREADS) -c $(SOFTWARE) > /dev/null'

bench-compress-software-l6-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SOFTWARE) \
		--level 6 --threads $(THREADS) --content-type software \
		--output $(BENCH_RESULTS_DIR)/compress-software-l6.json

bench-compress-software-l9: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: software L9 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -9 -p$(THREADS) -c $(SOFTWARE) > /dev/null'

bench-compress-software-l9-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(SOFTWARE) \
		--level 9 --threads $(THREADS) --content-type software \
		--output $(BENCH_RESULTS_DIR)/compress-software-l9.json

# --- Logs L1/L6/L9 ---
bench-compress-logs-l1: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: logs L1 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -1 -p$(THREADS) -c $(LOGS) > /dev/null'

bench-compress-logs-l1-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(LOGS) \
		--level 1 --threads $(THREADS) --content-type logs \
		--output $(BENCH_RESULTS_DIR)/compress-logs-l1.json

bench-compress-logs-l6: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: logs L6 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -6 -p$(THREADS) -c $(LOGS) > /dev/null'

bench-compress-logs-l6-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(LOGS) \
		--level 6 --threads $(THREADS) --content-type logs \
		--output $(BENCH_RESULTS_DIR)/compress-logs-l6.json

bench-compress-logs-l9: $(GZIPPY_BIN) bench-data
	@echo "=== Compression: logs L9 (gzippy only) ==="
	@time sh -c '$(GZIPPY_BIN) -9 -p$(THREADS) -c $(LOGS) > /dev/null'

bench-compress-logs-l9-all: bench-bin bench-data
	@mkdir -p $(BENCH_RESULTS_DIR)
	python3 scripts/benchmark_compression.py \
		--binaries $(BENCH_BIN_DIR) --data-file $(LOGS) \
		--level 9 --threads $(THREADS) --content-type logs \
		--output $(BENCH_RESULTS_DIR)/compress-logs-l9.json

# --- Combined compression ---
bench-compress: bench-compress-silesia-l6 bench-compress-software-l6 bench-compress-logs-l6

bench-compress-all: \
	bench-compress-silesia-l1-all bench-compress-silesia-l6-all bench-compress-silesia-l9-all \
	bench-compress-software-l1-all bench-compress-software-l6-all bench-compress-software-l9-all \
	bench-compress-logs-l1-all bench-compress-logs-l6-all bench-compress-logs-l9-all
	@echo ""
	@echo "=== Compression Results ==="
	@for f in $(BENCH_RESULTS_DIR)/compress-*.json; do \
		[ -f "$$f" ] && echo "--- $$(basename $$f .json) ---" && \
		python3 -c "import json; d=json.load(open('$$f')); [print(f'  {r[\"tool\"]}: {r.get(\"speed_mbps\",0):.1f} MB/s, ratio {r.get(\"ratio\",0):.3f}') for r in d.get('results',[]) if 'error' not in r]" 2>/dev/null || true; \
	done

# =============================================================================
# Combined benchmark targets
# =============================================================================

# Quick: gzippy only, L6, all datasets
bench: bench-decompress bench-compress
	@echo ""
	@echo "=== Quick Benchmark Complete ==="

# Full: all tools, all levels
bench-all: bench-decompress-all bench-compress-all
	@echo ""
	@echo "=== Full Benchmark Complete ==="
	@echo "Results in $(BENCH_RESULTS_DIR)/"

bench-exhaustive: bench-all

# =============================================================================
# Help
# =============================================================================
help:
	@echo "gzippy - The Fastest Parallel Gzip"
	@echo "======================================"
	@echo ""
	@echo "Quick commands (for AI tools and iteration):"
	@echo "  make              Build and run quick benchmark (< 30 seconds)"
	@echo "  make quick        Same as above"
	@echo "  make build        Build gzippy and ungzippy"
	@echo "  make deps         Build gzip and pigz from submodules"
	@echo "  make validate     Run validation suite (adaptive 3-17 trials)"
	@echo "  make lint         Run rustfmt and clippy (auto-fix)"
	@echo "  make lint-check   Check formatting without changes"
	@echo ""
	@echo "Benchmarks (gzippy only - fast):"
	@echo "  make bench                       Quick benchmark (L6, all datasets)"
	@echo "  make bench-decompress-silesia    Decompress silesia"
	@echo "  make bench-decompress-software   Decompress software"
	@echo "  make bench-decompress-logs       Decompress logs"
	@echo "  make bench-compress-silesia-l6   Compress silesia L6"
	@echo "  make bench-compress-software-l6  Compress software L6"
	@echo "  make bench-compress-logs-l6      Compress logs L6"
	@echo ""
	@echo "Benchmarks (all tools compared - exhaustive):"
	@echo "  make bench-all                       Full comparison (all tools, all levels)"
	@echo "  make bench-decompress-all            All decompression (3 datasets)"
	@echo "  make bench-decompress-silesia-all    Decompress silesia (all tools)"
	@echo "  make bench-decompress-software-all   Decompress software (all tools)"
	@echo "  make bench-decompress-logs-all       Decompress logs (all tools)"
	@echo "  make bench-compress-all              All compression (3 datasets x 3 levels)"
	@echo "  make bench-compress-silesia-l6-all   Compress silesia L6 (all tools)"
	@echo "  make bench-compress-software-l6-all  Compress software L6 (all tools)"
	@echo "  make bench-compress-logs-l6-all      Compress logs L6 (all tools)"
	@echo ""
	@echo "Data preparation:"
	@echo "  make bench-data   Prepare benchmark datasets (silesia, software, logs)"
	@echo ""
	@echo "Charting (separate test running from rendering):"
	@echo "  make validate-json     Run tests, save JSON to test_results/"
	@echo "  make render-chart      Generate charts from existing JSON (fast)"
	@echo "  make validation-chart  Both: run tests + generate charts"
	@echo ""
	@echo "Full testing (for humans at release time):"
	@echo "  make perf-full    			Comprehensive performance tests (10+ minutes)"
	@echo "  make test-data    			Generate all test data files"
	@echo ""
	@echo "Installation:"
	@echo "  make install      			Install gzippy and ungzippy to /usr/local/bin"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        			Remove all build artifacts and test data"
	@echo "  make help         			Show this message"
	@echo ""
	@echo "Binaries:"
	@echo "  gzippy              			Compress (default) or decompress with -d"
	@echo "  ungzippy            			Decompress (like gunzip/unpigz)"
