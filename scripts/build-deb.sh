#!/bin/bash
#
# Build Debian packages for rigz
#
# This script builds two packages:
#   - rigz: The main package with rigz and unrigz commands
#   - rigz-replace-gzip: Optional package that replaces system gzip
#
# Prerequisites:
#   sudo apt install debhelper cargo rustc devscripts
#
# Usage:
#   ./scripts/build-deb.sh
#
# Output:
#   ../rigz_0.1.0-1_amd64.deb
#   ../rigz-replace-gzip_0.1.0-1_all.deb

set -e

cd "$(dirname "$0")/.."

echo "=== Building rigz Debian packages ==="
echo ""

# Check for required tools
for cmd in dpkg-buildpackage cargo; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        echo "Install with: sudo apt install debhelper cargo rustc devscripts"
        exit 1
    fi
done

# Clean previous builds
echo "Cleaning previous builds..."
cargo clean 2>/dev/null || true
rm -rf debian/.debhelper debian/rigz debian/rigz-replace-gzip
rm -f debian/files debian/*.substvars debian/*.debhelper.log

# Build the packages
echo "Building packages..."
dpkg-buildpackage -us -uc -b

echo ""
echo "=== Build complete ==="
echo ""
echo "Packages created in parent directory:"
ls -la ../rigz*.deb 2>/dev/null || echo "  (no .deb files found)"
echo ""
echo "To install rigz only:"
echo "  sudo dpkg -i ../rigz_*.deb"
echo ""
echo "To also replace system gzip:"
echo "  sudo dpkg -i ../rigz_*.deb ../rigz-replace-gzip_*.deb"
echo ""
echo "To uninstall:"
echo "  sudo apt remove rigz-replace-gzip rigz"
