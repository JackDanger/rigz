# gzippy

**The fastest parallel gzip.** A drop-in replacement for gzip and pigz.

- **20-50% faster** compression than pigz at all levels
- **30-50% faster** decompression via libdeflate
- **100% compatible** with standard gzip

## Install

### From crates.io
```bash
cargo install gzippy
```

### From source
```bash
git clone --recursive https://github.com/jackdanger/gzippy
cd gzippy
cargo build --release
./target/release/gzippy --help
```

### Debian/Ubuntu
```bash
./scripts/build-deb.sh
sudo dpkg -i ../gzippy_*.deb

# Optional: Replace system gzip entirely
sudo dpkg -i ../gzippy-replace-gzip_*.deb
```

## Usage

```bash
gzippy file.txt           # Compress → file.txt.gz
gzippy -d file.txt.gz     # Decompress → file.txt
gzippy -p4 -9 file.txt    # 4 threads, best gzip-compatible compression
gzippy --ultra file.txt   # Near-zopfli compression (level 11)
gzippy --max file.txt     # Maximum compression (level 12)
cat data | gzippy > out   # Stdin → stdout
```

Same flags as gzip: `-1` to `-9`, `-c`, `-d`, `-k`, `-f`, `-r`, `-v`, `-q`.

Extended levels: `--level 10-12`, `--ultra` (L11), `--max` (L12).

## Performance

Benchmarked on 10MB text, 4 threads:

| Level | pigz | gzippy | Speedup |
|-------|------|--------|---------|
| L1 (fast) | 29ms | 24ms | **17% faster** |
| L6 (default) | 89ms | 76ms | **15% faster** |
| L9 (best) | 286ms | 201ms | **30% faster** |

Decompression is **30-50% faster** than pigz for all file types.

Output is byte-for-byte compatible with `gunzip`.

## How It Works

### Compression

**L1-L5**: Independent parallel blocks using [libdeflate](https://github.com/ebiggers/libdeflate)
- Each block is a complete gzip member
- Embeds block size markers for parallel decompression
- 30-50% faster than zlib

**L6-L9**: Pipelined compression using [zlib-ng](https://github.com/zlib-ng/zlib-ng)
- Dictionary sharing between blocks (like pigz)
- Dedicated writer thread for maximum throughput
- Matches pigz compression ratio

**L10-L12**: Ultra compression using libdeflate high levels
- Exhaustive search for near-zopfli compression
- 4-5% smaller output than gzip -9
- Slower than L9, but 10-20x faster than zopfli

### Decompression

Always uses libdeflate for maximum speed. Files compressed by gzippy with block markers decompress in parallel.

## Architecture

```
L1-L5:   Input → [Block₁] [Block₂] [Block₃] → libdeflate → Output
                      ↓        ↓        ↓
                 Independent blocks (parallel decompress)

L6-L9:   Input → [Block₁] → [Block₂] → [Block₃] → zlib-ng → Output
                      ↓           ↓           ↓
                 Dictionary chain (sequential decompress)

L10-L12: Input → [Block₁] [Block₂] [Block₃] → libdeflate L10-12 → Output
                      ↓        ↓        ↓
                 Ultra compression (parallel decompress, near-zopfli)
```

## Not a bgzip Replacement

gzippy uses BGZF-*style* concatenated gzip members with embedded block sizes, but it is **not compatible with bgzip** (from htslib/samtools):

- **Different subfield ID**: gzippy uses "GZ", bgzip uses "BC"
- **Different block sizes**: gzippy uses up to 128KB blocks, bgzip uses 64KB max
- **No BAM/tabix integration**: gzippy doesn't produce index-compatible output

Use bgzip for bioinformatics workflows that require indexed random access (BAM, VCF, etc.). Use gzippy for general-purpose fast parallel compression.

Both produce valid gzip output that any gzip tool can decompress.

## License

[zlib license](LICENSE) (same as zlib and pigz).

## Credits

- [pigz](https://zlib.net/pigz/) by Mark Adler - threading model inspiration
- [libdeflate](https://github.com/ebiggers/libdeflate) - fastest deflate implementation
- [zlib-ng](https://github.com/zlib-ng/zlib-ng) - SIMD-optimized zlib fork
