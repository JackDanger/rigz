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
UNRIGZ_BIN := $(RIGZ_DIR)/target/release/unrigz

.PHONY: all build quick perf-full test-data clean help validate deps

# =============================================================================
# Default target: quick benchmark for fast iteration (< 30 seconds)
# =============================================================================
all: quick

# =============================================================================
# Build targets
# =============================================================================

build: $(RIGZ_BIN) $(UNRIGZ_BIN)

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

# Create unrigz symlink (like unpigz)
$(UNRIGZ_BIN): $(RIGZ_BIN)
	@ln -sf rigz $(UNRIGZ_BIN)
	@echo "✓ Created unrigz symlink"

FORCE:

# =============================================================================
# Quick benchmark (~30 seconds) - for AI tools and fast iteration
# =============================================================================
quick: $(RIGZ_BIN) $(UNRIGZ_BIN) $(GZIP_BIN) $(PIGZ_BIN)
	@python3 scripts/perf.py --sizes 1,10 --levels 6 --threads 1,4

# =============================================================================
# Full performance tests (10+ minutes) - for humans at release time
# =============================================================================
perf-full: $(RIGZ_BIN) $(UNRIGZ_BIN) $(GZIP_BIN) $(PIGZ_BIN)
	@mkdir -p $(RESULTS_DIR)
	@python3 scripts/perf.py --full 2>&1 | tee $(RESULTS_DIR)/perf_full_$$(date +%Y%m%d_%H%M%S).log

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
# Validation target - cross-tool compression/decompression matrix
# =============================================================================
validate: $(RIGZ_BIN) $(UNRIGZ_BIN) $(GZIP_BIN) $(PIGZ_BIN)
	@python3 scripts/validate.py

# =============================================================================
# Install target
# =============================================================================
install: $(RIGZ_BIN) $(UNRIGZ_BIN)
	@echo "Installing to /usr/local/bin..."
	@install -m 755 $(RIGZ_BIN) /usr/local/bin/rigz
	@ln -sf rigz /usr/local/bin/unrigz
	@echo "✓ Installed rigz and unrigz"

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
	@echo "  make build        Build rigz and unrigz"
	@echo "  make deps         Build gzip and pigz from submodules"
	@echo "  make validate     Run validation suite"
	@echo ""
	@echo "Full testing (for humans at release time):"
	@echo "  make perf-full    Comprehensive performance tests (10+ minutes)"
	@echo "  make test-data    Generate all test data files"
	@echo ""
	@echo "Installation:"
	@echo "  make install      Install rigz and unrigz to /usr/local/bin"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Remove all build artifacts and test data"
	@echo "  make help         Show this message"
	@echo ""
	@echo "Binaries:"
	@echo "  rigz              Compress (default) or decompress with -d"
	@echo "  unrigz            Decompress (like gunzip/unpigz)"
