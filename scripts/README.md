# LLVM ASM Analysis Scripts

Tools for analyzing LLVM's generated assembly and closing the performance gap with our hand-written ASM.

## Quick Start

```bash
# 1. Generate LLVM assembly
RUSTFLAGS='--emit asm' cargo build --release

# 2. Extract and analyze LLVM's decode function
python3 scripts/llvm_to_inline_asm.py

# 3. Compare LLVM vs our v4 ASM
python3 scripts/compare_asm.py

# 4. Generate ready-to-use ASM patches
python3 scripts/generate_asm_patches.py
```

## Scripts

### `llvm_to_inline_asm.py`
Extracts LLVM's hot loop and converts it to Rust inline ASM format.

**Output:**
- `target/llvm_generated_full.rs` - Complete function with asm! block
- `target/llvm_raw_decode.s` - Raw LLVM assembly

### `compare_asm.py`
Compares LLVM's assembly with our v4 decoder, identifying:
- Instruction count differences
- Missing opcodes (BFXIL, CCMP, CSEL, etc.)
- Pattern differences

### `generate_asm_patches.py`
Generates specific ASM code patches that can be integrated into v4:
1. Optimized refill with BFXIL
2. Literal batch with preload
3. CCMP bounds check
4. Branchless refill with CSEL
5. Distance decode with BFXIL
6. Packed literal writes

**Output:**
- `target/asm_patches.rs` - Ready-to-use code snippets

## Key Findings

### LLVM's Advantages

1. **Heavy Unrolling**: 734 instructions vs our 157
2. **BFXIL**: Bitfield insert for `56 | (bitsleft & 7)`
3. **CCMP**: Chained conditions without extra branches
4. **CSEL**: Branchless value selection
5. **Constant registers**: Keeps 7, -1, 199 in registers

### Priority Optimizations

1. **BFXIL for bitsleft update** - Replaces OR with insert
2. **CCMP for bounds check** - Chains two conditions
3. **Preload pattern** - Hide memory latency
4. **Packed literal writes** - Write 2 bytes at once

## Register Mapping

LLVM's register allocation for `decode_huffman_libdeflate_style`:

| Register | Usage | Inline ASM |
|----------|-------|------------|
| x10 | in_pos | `{in_pos}` |
| x11 | bitbuf | `{bitbuf}` |
| x21 | bitsleft | `{bitsleft}` |
| x3 | out_pos | `{out_pos}` |
| x22 | entry | `{entry}` |
| x8 | in_ptr | `{in_ptr}` |
| x1 | out_ptr | `{out_ptr}` |
| x2 | out_len | `{out_len}` |
| x4 | litlen_ptr | `{litlen_ptr}` |
| x6 | dist_ptr | `{dist_ptr}` |
| x9 | in_end | `{in_end}` |
| x16 | constant 7 | literal |
| x17 | constant -1 | literal |
| x20 | constant 199 | literal |
| x12-x15, x19, x23-x28 | scratch | clobbers |

## Example: Applying BFXIL Patch

Before (v4):
```asm
"mov w14, #56",
"and w15, {bitsleft:w}, #7",
"orr {bitsleft:w}, w14, w15",
```

After (LLVM style):
```asm
"mov w14, #56",
"bfxil w14, {bitsleft:w}, #0, #3",  // Insert low 3 bits
"mov {bitsleft}, x14",
```

## Workflow

1. Run `compare_asm.py` to identify gaps
2. Run `generate_asm_patches.py` to get code snippets
3. Integrate patches into `src/asm_decode.rs`
4. Test: `cargo test --release test_asm_v4_correctness`
5. Benchmark: `cargo test --release bench_asm_v4_performance`
6. Iterate until performance matches or exceeds LLVM
