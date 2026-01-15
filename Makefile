# Rigz - Rust Parallel Gzip Replacement
# Build and test infrastructure
#
# Quick tests (<30s) run with 'make' or 'make quick' - for AI tools and iteration
# Full perf tests (10+ min) run with 'make perf-full' - for humans at release time

# Build configuration - submodules are in ./gzip and ./pigz
GZIP_DIR := ./gzip
PIGZ_DIR := ./pigz
RIGZ_DIR := .
TEST_DATA_DIR := test_data
RESULTS_DIR := test_results

# Build targets
GZIP_BIN := $(GZIP_DIR)/gzip
PIGZ_BIN := $(PIGZ_DIR)/pigz
RIGZ_BIN := $(RIGZ_DIR)/target/release/rigz

.PHONY: all build quick perf-full test-data clean help validate deps

# =============================================================================
# Default target: quick benchmark for fast iteration (< 30 seconds)
# =============================================================================
all: quick

# =============================================================================
# Build targets
# =============================================================================

build: $(RIGZ_BIN)

deps: $(GZIP_BIN) $(PIGZ_BIN)
	@echo "✓ Dependencies ready"

$(GZIP_BIN):
	@echo "Building gzip from source..."
	@cd $(GZIP_DIR) && ./configure --quiet 2>/dev/null || true
	@$(MAKE) -C $(GZIP_DIR) -j4 2>/dev/null || $(MAKE) -C $(GZIP_DIR)
	@echo "✓ Built gzip"

$(PIGZ_BIN):
	@echo "Building pigz from source..."
	@$(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 || true
	@$(MAKE) -C $(PIGZ_DIR) pigz >/dev/null 2>&1
	@echo "✓ Built pigz"

$(RIGZ_BIN): FORCE
	@echo "Building rigz..."
	@cd $(RIGZ_DIR) && cargo build --release 2>&1 | grep -E "(Compiling rigz|Finished|error)" || true
	@echo "✓ Built rigz"

FORCE:

# =============================================================================
# Quick benchmark (~20 seconds) - for AI tools and fast iteration
# =============================================================================
quick: $(RIGZ_BIN) $(GZIP_BIN) $(PIGZ_BIN)
	@chmod +x scripts/quick_bench.sh
	@./scripts/quick_bench.sh

# =============================================================================
# Full performance tests (10+ minutes) - for humans at release time
# =============================================================================
perf-full: $(RIGZ_BIN) $(GZIP_BIN) $(PIGZ_BIN) test-data
	@echo "============================================"
	@echo "  RIGZ Full Performance Suite"
	@echo "  (This will take 10+ minutes)"
	@echo "============================================"
	@mkdir -p $(RESULTS_DIR)
	@chmod +x scripts/perf_full.sh
	@./scripts/perf_full.sh 2>&1 | tee $(RESULTS_DIR)/perf_full_$$(date +%Y%m%d_%H%M%S).log

# Generate all test data files
test-data:
	@echo "Generating test data files..."
	@mkdir -p $(TEST_DATA_DIR)
	@[ -f $(TEST_DATA_DIR)/text-10KB.txt ] || head -c 10240 /dev/urandom | base64 > $(TEST_DATA_DIR)/text-10KB.txt 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/text-1MB.txt ] || head -c 1048576 /dev/urandom | base64 > $(TEST_DATA_DIR)/text-1MB.txt 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/text-10MB.txt ] || head -c 10485760 /dev/urandom | base64 > $(TEST_DATA_DIR)/text-10MB.txt 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/text-100MB.txt ] || head -c 104857600 /dev/urandom | base64 > $(TEST_DATA_DIR)/text-100MB.txt 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/random-10KB.dat ] || head -c 10240 /dev/urandom > $(TEST_DATA_DIR)/random-10KB.dat 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/random-1MB.dat ] || head -c 1048576 /dev/urandom > $(TEST_DATA_DIR)/random-1MB.dat 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/random-10MB.dat ] || head -c 10485760 /dev/urandom > $(TEST_DATA_DIR)/random-10MB.dat 2>/dev/null
	@[ -f $(TEST_DATA_DIR)/random-100MB.dat ] || head -c 104857600 /dev/urandom > $(TEST_DATA_DIR)/random-100MB.dat 2>/dev/null
	@echo "✓ Test data ready"

# =============================================================================
# Validation target - verify all outputs decompress correctly
# =============================================================================
validate: $(RIGZ_BIN)
	@echo "Running validation suite..."
	@mkdir -p $(TEST_DATA_DIR)
	@echo "test content for validation" > $(TEST_DATA_DIR)/validate.txt
	@passed=0; failed=0; \
	for level in 1 6 9; do \
		for threads in 1 4; do \
			$(RIGZ_BIN) -$$level -p$$threads -c $(TEST_DATA_DIR)/validate.txt > /tmp/v.gz 2>/dev/null; \
			if gunzip -c /tmp/v.gz 2>/dev/null | diff -q - $(TEST_DATA_DIR)/validate.txt >/dev/null 2>&1; then \
				echo "✓ Level $$level, $$threads thread(s)"; \
				passed=$$((passed+1)); \
			else \
				echo "✗ Level $$level, $$threads thread(s) FAILED"; \
				failed=$$((failed+1)); \
			fi; \
		done; \
	done; \
	rm -f /tmp/v.gz; \
	echo ""; \
	echo "Passed: $$passed, Failed: $$failed"; \
	[ $$failed -eq 0 ] || exit 1

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@echo "Cleaning..."
	@rm -rf $(TEST_DATA_DIR) $(RESULTS_DIR)
	@$(MAKE) -C $(PIGZ_DIR) clean >/dev/null 2>&1 || true
	@$(MAKE) -C $(GZIP_DIR) clean >/dev/null 2>&1 || true
	@cd $(RIGZ_DIR) && cargo clean >/dev/null 2>&1
	@echo "✓ Cleaned"

# =============================================================================
# Help
# =============================================================================
help:
	@echo "Rigz - Rust Parallel Gzip Replacement"
	@echo "======================================"
	@echo ""
	@echo "Quick commands (for AI tools and iteration):"
	@echo "  make              Build and run quick benchmark (< 30 seconds)"
	@echo "  make quick        Same as above"
	@echo "  make build        Build rigz only"
	@echo "  make deps         Build gzip and pigz from submodules"
	@echo "  make validate     Run validation suite"
	@echo ""
	@echo "Full testing (for humans at release time):"
	@echo "  make perf-full    Comprehensive performance tests (10+ minutes)"
	@echo "  make test-data    Generate all test data files"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Remove all build artifacts and test data"
	@echo "  make help         Show this message"
	@echo ""
	@echo "Performance targets:"
	@echo "  - Beat gzip single-threaded (within 5%)"
	@echo "  - Beat pigz multi-threaded"
	@echo "  - Valid gzip output that works with gunzip"
