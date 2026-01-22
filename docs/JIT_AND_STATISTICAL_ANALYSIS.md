# JIT and Statistical Modeling Analysis for gzippy

## Current Performance Status
- **Current**: 93-98% of libdeflate (variance due to system noise)
- **Peak achieved**: 98.2% of libdeflate (1422 MB/s vs 1449 MB/s)
- **Cache hit rate**: 90% on fingerprints (~100 unique tables in Silesia)
- **Specialized decoder**: Only 3.6% usage (requires all codes ≤11 bits)

## Key Insights from Source Analysis

### libdeflate (decompress_template.h)
1. **CAN_CONSUME_AND_THEN_PRELOAD**: Compile-time bit budget verification
2. **Up to 2 extra fast literals**: After first literal, try 2 more before refill
3. **Preload before write**: `entry = table[bitbuf]; *out++ = lit;` order
4. **EXTRACT_VARBITS8**: Cast to u8 before shift for performance
5. **saved_bitbuf pattern**: Save bits BEFORE consuming for extra bit extraction

### rapidgzip (HuffmanCodingShortBitsMultiCached.hpp)
1. **CacheEntry (4 bytes)**: Packed `needToReadDistanceBits:1 | bitsToSkip:6 | symbolCount:2 | symbols:18`
2. **DISTANCE_OFFSET = 254**: Combine length+distance in symbol space
3. **Pre-expanded length codes**: If `codeLength + extraBits <= LUT_BITS`, expand all combinations
4. **alignas(64)**: Cache line alignment for LUT

---

## Four JIT Implementation Approaches

### 1. Cranelift Runtime Code Generation
**Concept**: Use Rust's Cranelift JIT to generate native machine code for each unique fingerprint.

**Implementation**:
```rust
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};

struct JITDecoder {
    module: JITModule,
    // Map fingerprint -> compiled function pointer
    decoders: HashMap<TableFingerprint, fn(&mut Bits, &mut [u8]) -> usize>,
}

impl JITDecoder {
    fn compile_for_fingerprint(&mut self, litlen_lens: &[u8]) -> fn(...) {
        let mut ctx = self.module.make_context();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_ctx);

        // For each symbol in the table, generate:
        // - Immediate load of symbol value (no memory access)
        // - Constant bit width (no variable shifts)
        // - Direct branch to next state

        for (bits_pattern, symbol, bits_consumed) in table_entries {
            // Generate: if (bitbuf & mask) == bits_pattern { emit symbol }
            let block = builder.create_block();
            builder.switch_to_block(block);
            // ... emit IR
        }

        self.module.define_function(func_id, &mut ctx)?;
        self.module.finalize_definitions();
        self.module.get_finalized_function(func_id)
    }
}
```

**Pros**: True native code, eliminates table lookups entirely
**Cons**: ~100μs compilation per fingerprint, Cranelift dependency (~2MB)
**Expected gain**: 5-15% for repeated fingerprints

---

### 2. Threaded Bytecode Interpreter
**Concept**: Generate specialized bytecode at table-build time, execute with computed goto.

**Implementation**:
```rust
#[repr(u8)]
enum DecodeOp {
    Literal { value: u8, bits: u8 },
    Length { base: u16, extra: u8, bits: u8 },
    EndOfBlock { bits: u8 },
    Subtable { offset: u16, bits: u8 },
}

struct BytecodeDecoder {
    ops: Vec<DecodeOp>,  // Indexed by 11-bit pattern
}

#[inline(never)]
fn decode_bytecode(bc: &BytecodeDecoder, bits: &mut Bits, out: &mut [u8]) -> usize {
    let mut out_pos = 0;
    loop {
        let op = &bc.ops[(bits.peek() & 0x7FF) as usize];
        match op {
            DecodeOp::Literal { value, bits: n } => {
                out[out_pos] = *value;
                out_pos += 1;
                bits.consume(*n as u32);
            }
            DecodeOp::EndOfBlock { bits: n } => {
                bits.consume(*n as u32);
                return out_pos;
            }
            // ...
        }
    }
}
```

**Pros**: No JIT compilation latency, portable, simple
**Cons**: Match dispatch overhead (~2-5x slower than direct table)
**Expected gain**: -5% to +2% (interpretation overhead may exceed benefit)

---

### 3. Build-Time Proc Macro Code Generation
**Concept**: Analyze datasets offline, generate Rust code for top fingerprints at compile time.

**Implementation**:
```rust
// In build.rs or proc macro:
// 1. Analyze silesia, enwik8, calgary to find common fingerprints
// 2. For each fingerprint, generate a specialized decode function

// Generated code (specialized_fingerprints.rs):
#[inline(never)]
fn decode_fingerprint_0x1234567890abcdef(bits: &mut Bits, out: &mut [u8]) -> usize {
    let mut out_pos = 0;
    loop {
        let b = bits.peek();
        // Hardcoded decision tree for this specific table
        if b & 0xFF == 0x30 { out[out_pos] = b'e'; out_pos += 1; bits.consume(8); continue; }
        if b & 0x1FF == 0x130 { out[out_pos] = b't'; out_pos += 1; bits.consume(9); continue; }
        // ... more patterns
    }
}

// Dispatcher:
fn decode_by_fingerprint(fp: u64, bits: &mut Bits, out: &mut [u8]) -> Option<usize> {
    match fp {
        0x1234567890abcdef => Some(decode_fingerprint_0x1234567890abcdef(bits, out)),
        0xfedcba0987654321 => Some(decode_fingerprint_fedcba0987654321(bits, out)),
        // Top 50 fingerprints
        _ => None, // Fall back to generic decoder
    }
}
```

**Pros**: Zero runtime compilation, statically optimized by LLVM
**Cons**: Only covers pre-analyzed fingerprints, binary size growth
**Expected gain**: 10-20% for known fingerprints (0% for unknown)

---

### 4. LLVM IR Generation with Runtime Compilation
**Concept**: Generate LLVM IR at runtime, leverage LLVM's optimizer for best code quality.

**Implementation**:
```rust
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;

fn compile_decoder_llvm(litlen_lens: &[u8]) -> unsafe fn(*mut Bits, *mut u8) -> usize {
    let context = Context::create();
    let module = context.create_module("huffman_decoder");
    let builder = context.create_builder();

    // Build LLVM function
    let fn_type = context.void_type().fn_type(&[...], false);
    let function = module.add_function("decode", fn_type, None);

    // Generate optimal decision tree as LLVM IR
    // LLVM will optimize this into jump tables, constant propagation, etc.

    let engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;
    engine.get_function::<unsafe fn(...)>("decode")
}
```

**Pros**: Best possible code quality, same as static compilation
**Cons**: Heavy dependency (~50MB), 10-50ms compilation time per table
**Expected gain**: 15-25% for repeated fingerprints

---

## Four Statistical Modeling Approaches

### 1. Markov Chain Symbol Prediction
**Concept**: Use P(symbol[i+1] | symbol[i]) to predict and speculatively decode next symbol.

**Implementation**:
```rust
struct MarkovPredictor {
    // P(next | current) as most-likely-next for each literal
    // Index by current literal (0-255), value is predicted next
    prediction: [u8; 256],
    // Confidence threshold
    confidence: [u8; 256],  // 0-255 scaled probability
}

impl MarkovPredictor {
    fn train(data: &[u8]) {
        let mut counts = [[0u32; 256]; 256];
        for window in data.windows(2) {
            counts[window[0] as usize][window[1] as usize] += 1;
        }
        for i in 0..256 {
            let (max_j, max_count) = counts[i].iter().enumerate()
                .max_by_key(|(_, &c)| c).unwrap();
            self.prediction[i] = max_j as u8;
            let total: u32 = counts[i].iter().sum();
            self.confidence[i] = ((max_count * 255) / total.max(1)) as u8;
        }
    }
}

// In hot loop after decoding literal `lit`:
if predictor.confidence[lit as usize] > 200 {
    let predicted = predictor.prediction[lit as usize];
    let predicted_entry = litlen[bits.peek() & 0x7FF];
    if (predicted_entry.raw() as i32) < 0 && predicted_entry.literal_value() == predicted {
        // Prediction correct! Accept both symbols
        out[out_pos] = lit;
        out[out_pos + 1] = predicted;
        out_pos += 2;
        bits.consume_entry(predicted_entry.raw());
        continue;
    }
}
```

**Training data**: After 'e' → ' ' (20%), 's' (15%), 'r' (12%), 'd' (10%)
**Pros**: No extra table lookup when prediction correct
**Cons**: Memory for 256-byte tables, branch prediction may suffer
**Expected gain**: 3-8% on text-heavy data, 0% on binary

---

### 2. Adaptive Table Sizing (Dynamic litlen_tablebits)
**Concept**: Use smaller tables when code lengths allow, improving cache utilization.

**Implementation**:
```rust
struct AdaptiveTable {
    // Tables at different sizes
    table_10bit: Option<Box<[Entry; 1024]>>,   // 4KB
    table_11bit: Option<Box<[Entry; 2048]>>,   // 8KB
    table_12bit: Option<Box<[Entry; 4096]>>,   // 16KB
    active_bits: u8,
    mask: u64,
}

impl AdaptiveTable {
    fn build(lengths: &[u8]) -> Self {
        let max_len = lengths.iter().copied().filter(|&l| l > 0).max().unwrap_or(0);

        if max_len <= 10 {
            // Build 10-bit table (no subtables needed)
            Self { table_10bit: Some(build_10bit(lengths)), active_bits: 10, mask: 0x3FF, .. }
        } else if max_len <= 11 {
            Self { table_11bit: Some(build_11bit(lengths)), active_bits: 11, mask: 0x7FF, .. }
        } else {
            // Need 12-bit or subtables
            Self { table_12bit: Some(build_12bit(lengths)), active_bits: 12, mask: 0xFFF, .. }
        }
    }

    #[inline(always)]
    fn lookup(&self, bits: u64) -> Entry {
        match self.active_bits {
            10 => self.table_10bit.as_ref().unwrap()[(bits & 0x3FF) as usize],
            11 => self.table_11bit.as_ref().unwrap()[(bits & 0x7FF) as usize],
            12 => self.table_12bit.as_ref().unwrap()[(bits & 0xFFF) as usize],
            _ => unreachable!(),
        }
    }
}
```

**Pros**: Better L1 cache utilization for simple codes
**Cons**: Runtime dispatch overhead, code complexity
**Expected gain**: 2-5% on data with simple Huffman codes

---

### 3. Run-Length Detection and SIMD Parallel Decode
**Concept**: When consecutive symbols have same code length, decode multiple in parallel with SIMD.

**Implementation**:
```rust
#[cfg(target_arch = "x86_64")]
unsafe fn decode_uniform_literals_avx2(
    bits: u64,
    code_len: u8,  // Uniform code length (e.g., 8)
    count: usize,  // Number to decode (e.g., 8)
    symbol_table: &[u8; 256],  // Direct symbol lookup
) -> ([u8; 8], u32) {
    use std::arch::x86_64::*;

    // Extract 8 indices of `code_len` bits each
    let mask = (1u64 << code_len) - 1;
    let mut indices = [0u8; 8];
    for i in 0..8 {
        indices[i] = ((bits >> (i * code_len as usize)) & mask) as u8;
    }

    // Parallel table lookup using vpshufb
    let idx_vec = _mm_loadu_si64(indices.as_ptr() as *const _);
    let table_vec = _mm_loadu_si128(symbol_table.as_ptr() as *const __m128i);
    let result = _mm_shuffle_epi8(table_vec, idx_vec);

    let mut symbols = [0u8; 8];
    _mm_storeu_si64(symbols.as_mut_ptr() as *mut _, result);

    (symbols, (code_len as u32) * 8)
}

// Detection in hot loop:
if current_entry.is_8bit_literal() {
    let peek = bits.peek();
    // Check if next 7 entries are also 8-bit literals
    if is_uniform_8bit_run(peek, litlen_table) {
        let (symbols, bits_consumed) = decode_uniform_literals_avx2(...);
        out[out_pos..out_pos+8].copy_from_slice(&symbols);
        out_pos += 8;
        bits.consume(bits_consumed);
        continue;
    }
}
```

**Pros**: Up to 8x throughput for uniform runs
**Cons**: Detection overhead, only helps specific patterns
**Expected gain**: 10-30% on data with long literal runs (PNG, JSON)

---

### 4. Block-Level Content Classification
**Concept**: Classify deflate blocks and apply block-type-specific optimizations.

**Implementation**:
```rust
#[derive(Clone, Copy)]
enum BlockType {
    LiteralHeavy,  // >70% literals (text, JSON)
    MatchHeavy,    // >50% matches (binary, already-compressed)
    Mixed,         // Balanced
}

struct BlockClassifier {
    // Per-fingerprint classification (learned)
    fingerprint_types: HashMap<TableFingerprint, BlockType>,
}

impl BlockClassifier {
    fn classify(&mut self, fp: TableFingerprint, litlen_lens: &[u8]) -> BlockType {
        if let Some(&t) = self.fingerprint_types.get(&fp) {
            return t;
        }

        // Heuristic: count non-zero lengths in literal range vs length range
        let literal_codes: usize = litlen_lens[0..256].iter().filter(|&&l| l > 0).count();
        let length_codes: usize = litlen_lens[257..286].iter().filter(|&&l| l > 0).count();

        let block_type = if literal_codes > 200 && length_codes < 10 {
            BlockType::LiteralHeavy
        } else if length_codes > 20 {
            BlockType::MatchHeavy
        } else {
            BlockType::Mixed
        };

        self.fingerprint_types.insert(fp, block_type);
        block_type
    }
}

// In decoder dispatch:
match classifier.classify(fingerprint, litlen_lens) {
    BlockType::LiteralHeavy => decode_literal_optimized(bits, out, litlen, dist),
    BlockType::MatchHeavy => decode_match_optimized(bits, out, litlen, dist),
    BlockType::Mixed => decode_standard(bits, out, litlen, dist),
}
```

**Literal-heavy optimizations**:
- Decode 5+ literals per iteration
- Skip distance table preload

**Match-heavy optimizations**:
- Prefetch match source memory
- Use larger copy buffers

**Pros**: Tailored decode path per content type
**Cons**: Classification overhead, maintenance complexity
**Expected gain**: 5-10% when classification accurate

---

## Recommended Implementation Order

1. **Build-Time Proc Macro JIT** (#3) - Low risk, measurable gain
2. **Run-Length SIMD** (#3 statistical) - High gain for specific data
3. **Markov Prediction** (#1 statistical) - Interesting experiment
4. **Cranelift JIT** (#1 JIT) - High effort, high potential reward

## Benchmarking Strategy

For each approach:
1. Test on Silesia (mixed), enwik8 (text), calgary (binary)
2. Measure both throughput AND variance
3. Track regression with CI
4. Document results in CLAUDE.md
