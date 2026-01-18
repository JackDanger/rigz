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
	@rm -rf $(TEST_DATA_DIR) $(RESULTS_DIR)
	@$(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 || true
	@$(MAKE) -C $(GZIP_DIR) clean >/dev/null 2>&1 || true
	@cd $(GZIPPY_DIR) && cargo clean >/dev/null 2>&1
	@echo "✓ Cleaned"

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
