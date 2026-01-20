#!/usr/bin/env bash
# =============================================================================
# Build all benchmark tools for CI
# =============================================================================
# This script builds gzippy and all competitor tools used in CI benchmarks.
# It's shared across multiple workflows to ensure consistency.
#
# Usage: ./scripts/build-tools.sh [--gzippy] [--pigz] [--rapidgzip] [--igzip] [--zopfli] [--libdeflate] [--gzip]
#        ./scripts/build-tools.sh --all
#
# Output paths (relative to repo root):
#   gzippy:    target/release/gzippy
#   pigz:      pigz/pigz, pigz/unpigz
#   rapidgzip: rapidgzip/librapidarchive/build/src/tools/rapidgzip
#   igzip:     isa-l/build/igzip
#   zopfli:    zopfli/zopfli
#   libdeflate: libdeflate/build/programs/libdeflate-gzip
#   gzip:      gzip/gzip, gzip/gunzip
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Parse arguments
BUILD_GZIPPY=false
BUILD_PIGZ=false
BUILD_RAPIDGZIP=false
BUILD_IGZIP=false
BUILD_ZOPFLI=false
BUILD_LIBDEFLATE=false
BUILD_GZIP=false

if [[ $# -eq 0 ]]; then
    log_error "Usage: $0 [--gzippy] [--pigz] [--rapidgzip] [--igzip] [--zopfli] [--libdeflate] [--gzip] [--all]"
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --gzippy) BUILD_GZIPPY=true ;;
        --pigz) BUILD_PIGZ=true ;;
        --rapidgzip) BUILD_RAPIDGZIP=true ;;
        --igzip) BUILD_IGZIP=true ;;
        --zopfli) BUILD_ZOPFLI=true ;;
        --libdeflate) BUILD_LIBDEFLATE=true ;;
        --gzip) BUILD_GZIP=true ;;
        --all)
            BUILD_GZIPPY=true
            BUILD_PIGZ=true
            BUILD_RAPIDGZIP=true
            BUILD_IGZIP=true
            BUILD_ZOPFLI=true
            BUILD_LIBDEFLATE=true
            BUILD_GZIP=true
            ;;
        *)
            log_error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

# =============================================================================
# Build gzippy
# =============================================================================
if $BUILD_GZIPPY; then
    log_info "Building gzippy..."
    cargo build --release
    
    if [[ -f target/release/gzippy ]]; then
        log_info "✓ gzippy built: target/release/gzippy"
        # Verify binary is self-contained
        if command -v ldd &> /dev/null; then
            ldd target/release/gzippy 2>/dev/null || true
        elif command -v otool &> /dev/null; then
            otool -L target/release/gzippy || true
        fi
    else
        log_error "gzippy build failed"
        exit 1
    fi
fi

# =============================================================================
# Build pigz
# =============================================================================
if $BUILD_PIGZ; then
    log_info "Building pigz..."
    make -C pigz -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    
    if [[ -f pigz/pigz && -f pigz/unpigz ]]; then
        log_info "✓ pigz built: pigz/pigz, pigz/unpigz"
    else
        log_error "pigz build failed"
        exit 1
    fi
fi

# =============================================================================
# Build rapidgzip
# =============================================================================
if $BUILD_RAPIDGZIP; then
    log_info "Building rapidgzip..."
    
    cd "$REPO_ROOT/rapidgzip/librapidarchive"
    rm -rf build
    mkdir -p build
    cd build
    
    # Build without ISA-L for now (simpler, works cross-platform)
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_ISAL=OFF
    make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" rapidgzip
    
    cd "$REPO_ROOT"
    
    # The binary is at build/src/tools/rapidgzip (note: src/tools, not tools)
    RAPIDGZIP_BIN="rapidgzip/librapidarchive/build/src/tools/rapidgzip"
    if [[ -f "$RAPIDGZIP_BIN" ]]; then
        log_info "✓ rapidgzip built: $RAPIDGZIP_BIN"
        # Verify it runs
        "$RAPIDGZIP_BIN" --version || log_warn "rapidgzip --version failed"
    else
        log_error "rapidgzip build failed - binary not found at $RAPIDGZIP_BIN"
        # Debug: show what exists
        log_info "Contents of build directory:"
        find rapidgzip/librapidarchive/build -name "rapidgzip" -type f 2>/dev/null || true
        exit 1
    fi
fi

# =============================================================================
# Build igzip (ISA-L)
# =============================================================================
if $BUILD_IGZIP; then
    log_info "Building igzip (ISA-L)..."
    
    cd "$REPO_ROOT/isa-l"
    rm -rf build
    mkdir -p build
    cd build
    
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" igzip || {
        log_warn "igzip build failed (may be expected on some platforms)"
        cd "$REPO_ROOT"
    }
    
    cd "$REPO_ROOT"
    
    # Binary is at build/igzip (not build/bin/igzip)
    IGZIP_BIN="isa-l/build/igzip"
    if [[ -f "$IGZIP_BIN" ]]; then
        log_info "✓ igzip built: $IGZIP_BIN"
    else
        log_warn "igzip not built - may be unavailable on this platform"
    fi
fi

# =============================================================================
# Build zopfli
# =============================================================================
if $BUILD_ZOPFLI; then
    log_info "Building zopfli..."
    make -C zopfli zopfli -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    
    if [[ -f zopfli/zopfli ]]; then
        log_info "✓ zopfli built: zopfli/zopfli"
    else
        log_error "zopfli build failed"
        exit 1
    fi
fi

# =============================================================================
# Build libdeflate
# =============================================================================
if $BUILD_LIBDEFLATE; then
    log_info "Building libdeflate..."
    
    cd "$REPO_ROOT/libdeflate"
    rm -rf build
    mkdir -p build
    cd build
    
    cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBDEFLATE_BUILD_SHARED_LIB=OFF
    make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    
    cd "$REPO_ROOT"
    
    LIBDEFLATE_BIN="libdeflate/build/programs/libdeflate-gzip"
    if [[ -f "$LIBDEFLATE_BIN" ]]; then
        log_info "✓ libdeflate built: $LIBDEFLATE_BIN"
        "$LIBDEFLATE_BIN" -h 2>&1 | head -1 || true
    else
        log_error "libdeflate build failed - binary not found at $LIBDEFLATE_BIN"
        find libdeflate/build -name "libdeflate*" -type f 2>/dev/null || true
        exit 1
    fi
fi

# =============================================================================
# Build gzip (GNU gzip)
# =============================================================================
if $BUILD_GZIP; then
    log_info "Building gzip..."
    
    cd "$REPO_ROOT/gzip"
    
    # gzip uses autotools - check if we need to configure
    if [[ ! -f Makefile ]]; then
        if [[ -f configure ]]; then
            ./configure --prefix="$REPO_ROOT/gzip/install" || {
                log_warn "gzip configure failed - may need autotools"
                cd "$REPO_ROOT"
            }
        else
            log_warn "gzip configure not found - trying autoreconf"
            autoreconf -i 2>/dev/null || log_warn "autoreconf not available"
            if [[ -f configure ]]; then
                ./configure --prefix="$REPO_ROOT/gzip/install" || {
                    log_warn "gzip configure failed"
                    cd "$REPO_ROOT"
                }
            fi
        fi
    fi
    
    if [[ -f Makefile ]]; then
        make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" || log_warn "gzip make failed"
    fi
    
    cd "$REPO_ROOT"
    
    if [[ -f gzip/gzip ]]; then
        log_info "✓ gzip built: gzip/gzip"
        # Create gunzip symlink if missing
        if [[ ! -f gzip/gunzip ]]; then
            ln -sf gzip gzip/gunzip
        fi
    else
        log_warn "gzip not built - may need system gzip instead"
    fi
fi

log_info "Build complete!"
