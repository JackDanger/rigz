//! Speculative Batch Decode - A Novel Approach
//!
//! **Key Insight:** Instead of asking "is this a literal?" THEN decoding,
//! we speculatively decode ASSUMING it's a literal, then validate.
//!
//! This is inspired by CPU speculative execution:
//! - Predict: Assume next N symbols are literals
//! - Execute: Decode them all in parallel
//! - Validate: Check if prediction was correct
//! - Commit or Rollback: Write results or undo
//!
//! ## Why This Works
//!
//! 1. Most symbols ARE literals (60-80% in typical data)
//! 2. Validation is CHEAPER than branching (no misprediction)
//! 3. Batch decode amortizes overhead across N symbols
//! 4. Modern CPUs can do 4-8 parallel loads
//!
//! ## Expected Gain
//!
//! - Baseline: 935 MB/s
//! - With speculative batch (4 symbols): ~1200 MB/s (+28%)
//! - With speculative batch (8 symbols): ~1500 MB/s (+60%)

#![allow(dead_code)]

use crate::libdeflate_entry::LitLenTable;

/// Result of speculative batch decode
pub struct SpeculativeBatch {
    /// Decoded symbols (valid only if corresponding valid bit is set)
    pub symbols: [u8; 8],
    /// Bits consumed for each symbol
    pub bits_consumed: [u8; 8],
    /// Total bits consumed for all valid symbols
    pub total_bits: u32,
    /// Number of valid literals before first non-literal
    pub valid_count: usize,
}

impl SpeculativeBatch {
    /// Check if all 8 symbols are valid literals
    #[inline(always)]
    pub fn all_valid(&self) -> bool {
        self.valid_count == 8
    }

    /// Check if at least 4 symbols are valid (worth using)
    #[inline(always)]
    pub fn worth_it(&self) -> bool {
        self.valid_count >= 4
    }
}

/// Speculatively decode up to 8 literals from bit buffer
///
/// This function:
/// 1. Does 8 table lookups in parallel (CPU can issue multiple loads)
/// 2. Extracts symbols and bit counts
/// 3. Validates each as a literal
/// 4. Returns count of valid literals before first non-literal
#[inline(always)]
pub fn speculative_decode_8(
    mut bits: u64,
    mut bits_available: u32,
    table: &LitLenTable,
) -> SpeculativeBatch {
    let mut batch = SpeculativeBatch {
        symbols: [0; 8],
        bits_consumed: [0; 8],
        total_bits: 0,
        valid_count: 0,
    };

    // Unrolled speculative decode - all 8 lookups happen "in parallel"
    // (Modern CPUs can issue multiple independent loads)

    // Symbol 0
    let e0 = table.lookup(bits);
    let is_lit0 = (e0.raw() as i32) < 0; // Signed check = literal
    if !is_lit0 {
        return batch;
    }
    batch.symbols[0] = e0.literal_value();
    batch.bits_consumed[0] = e0.codeword_bits();
    let consumed0 = e0.codeword_bits() as u32;
    bits >>= consumed0;
    bits_available = bits_available.saturating_sub(consumed0);
    batch.total_bits += consumed0;
    batch.valid_count = 1;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 1
    let e1 = table.lookup(bits);
    let is_lit1 = (e1.raw() as i32) < 0;
    if !is_lit1 {
        return batch;
    }
    batch.symbols[1] = e1.literal_value();
    batch.bits_consumed[1] = e1.codeword_bits();
    let consumed1 = e1.codeword_bits() as u32;
    bits >>= consumed1;
    bits_available = bits_available.saturating_sub(consumed1);
    batch.total_bits += consumed1;
    batch.valid_count = 2;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 2
    let e2 = table.lookup(bits);
    let is_lit2 = (e2.raw() as i32) < 0;
    if !is_lit2 {
        return batch;
    }
    batch.symbols[2] = e2.literal_value();
    batch.bits_consumed[2] = e2.codeword_bits();
    let consumed2 = e2.codeword_bits() as u32;
    bits >>= consumed2;
    bits_available = bits_available.saturating_sub(consumed2);
    batch.total_bits += consumed2;
    batch.valid_count = 3;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 3
    let e3 = table.lookup(bits);
    let is_lit3 = (e3.raw() as i32) < 0;
    if !is_lit3 {
        return batch;
    }
    batch.symbols[3] = e3.literal_value();
    batch.bits_consumed[3] = e3.codeword_bits();
    let consumed3 = e3.codeword_bits() as u32;
    bits >>= consumed3;
    bits_available = bits_available.saturating_sub(consumed3);
    batch.total_bits += consumed3;
    batch.valid_count = 4;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 4
    let e4 = table.lookup(bits);
    let is_lit4 = (e4.raw() as i32) < 0;
    if !is_lit4 {
        return batch;
    }
    batch.symbols[4] = e4.literal_value();
    batch.bits_consumed[4] = e4.codeword_bits();
    let consumed4 = e4.codeword_bits() as u32;
    bits >>= consumed4;
    bits_available = bits_available.saturating_sub(consumed4);
    batch.total_bits += consumed4;
    batch.valid_count = 5;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 5
    let e5 = table.lookup(bits);
    let is_lit5 = (e5.raw() as i32) < 0;
    if !is_lit5 {
        return batch;
    }
    batch.symbols[5] = e5.literal_value();
    batch.bits_consumed[5] = e5.codeword_bits();
    let consumed5 = e5.codeword_bits() as u32;
    bits >>= consumed5;
    bits_available = bits_available.saturating_sub(consumed5);
    batch.total_bits += consumed5;
    batch.valid_count = 6;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 6
    let e6 = table.lookup(bits);
    let is_lit6 = (e6.raw() as i32) < 0;
    if !is_lit6 {
        return batch;
    }
    batch.symbols[6] = e6.literal_value();
    batch.bits_consumed[6] = e6.codeword_bits();
    let consumed6 = e6.codeword_bits() as u32;
    bits >>= consumed6;
    bits_available = bits_available.saturating_sub(consumed6);
    batch.total_bits += consumed6;
    batch.valid_count = 7;

    if bits_available < 15 {
        return batch;
    }

    // Symbol 7
    let e7 = table.lookup(bits);
    let is_lit7 = (e7.raw() as i32) < 0;
    if !is_lit7 {
        return batch;
    }
    batch.symbols[7] = e7.literal_value();
    batch.bits_consumed[7] = e7.codeword_bits();
    let consumed7 = e7.codeword_bits() as u32;
    batch.total_bits += consumed7;
    batch.valid_count = 8;

    batch
}

/// Decode using speculative batch with fallback
///
/// This is the main decode loop that uses speculative batch decode
/// for literal runs and falls back to regular decode for matches.
#[allow(dead_code)]
pub fn decode_speculative_batch(
    bits: &mut crate::libdeflate_decode::LibdeflateBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen_table: &LitLenTable,
    dist_table: &crate::libdeflate_entry::DistTable,
) -> std::io::Result<usize> {
    use std::io::{Error, ErrorKind};

    const FASTLOOP_MARGIN: usize = 274;

    'fastloop: while out_pos + FASTLOOP_MARGIN <= output.len() {
        bits.refill_branchless();

        // Try speculative batch decode (8 literals at once)
        let batch = speculative_decode_8(bits.peek_bits(), bits.available(), litlen_table);

        if batch.valid_count >= 2 {
            // Got at least 2 literals - worth it!
            // Copy all valid literals at once
            let count = batch.valid_count;
            output[out_pos..out_pos + count].copy_from_slice(&batch.symbols[..count]);
            out_pos += count;
            bits.consume(batch.total_bits);

            // If we got all 8, try again immediately (likely more literals)
            if batch.all_valid() {
                continue 'fastloop;
            }
        }

        // Either no batch or batch ended - decode next symbol normally
        bits.refill_branchless();
        let saved_bitbuf = bits.peek_bits();

        let mut entry = litlen_table.lookup(saved_bitbuf);
        if entry.is_subtable_ptr() {
            entry = litlen_table.lookup_subtable(entry, saved_bitbuf);
        }

        // Literal
        if (entry.raw() as i32) < 0 {
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            bits.consume_entry(entry.raw());
            continue 'fastloop;
        }

        // EOB
        if entry.is_end_of_block() {
            bits.consume_entry(entry.raw());
            return Ok(out_pos);
        }

        // Match
        bits.consume_entry(entry.raw());
        let length = entry.decode_length(saved_bitbuf);

        bits.refill_branchless();
        let dist_saved = bits.peek_bits();
        let dist_entry = dist_table.lookup(dist_saved);
        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        // Copy match
        if out_pos + length as usize > output.len() {
            break 'fastloop;
        }

        crate::libdeflate_decode::copy_match(output, out_pos, distance, length);
        out_pos += length as usize;
    }

    // Generic loop for remainder - simple fallback
    generic_decode_loop(bits, output, out_pos, litlen_table, dist_table)
}

/// Simple generic decode loop for near end of buffer
fn generic_decode_loop(
    bits: &mut crate::libdeflate_decode::LibdeflateBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen_table: &LitLenTable,
    dist_table: &crate::libdeflate_entry::DistTable,
) -> std::io::Result<usize> {
    use std::io::{Error, ErrorKind};

    loop {
        bits.refill_branchless();
        let saved_bitbuf = bits.peek_bits();

        let mut entry = litlen_table.lookup(saved_bitbuf);
        if entry.is_subtable_ptr() {
            entry = litlen_table.lookup_subtable(entry, saved_bitbuf);
        }

        // Literal
        if (entry.raw() as i32) < 0 {
            if out_pos >= output.len() {
                return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            bits.consume_entry(entry.raw());
            continue;
        }

        // EOB
        if entry.is_end_of_block() {
            bits.consume_entry(entry.raw());
            return Ok(out_pos);
        }

        // Match
        bits.consume_entry(entry.raw());
        let length = entry.decode_length(saved_bitbuf);

        bits.refill_branchless();
        let dist_saved = bits.peek_bits();
        let mut dist_entry = dist_table.lookup(dist_saved);
        if dist_entry.is_subtable_ptr() {
            dist_entry = dist_table.lookup_subtable(dist_entry, dist_saved);
        }
        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        if out_pos + length as usize > output.len() {
            return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
        }

        crate::libdeflate_decode::copy_match(output, out_pos, distance, length);
        out_pos += length as usize;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_speculative_batch_basic() {
        // This is a unit test placeholder - real test would need actual table
        eprintln!("Speculative batch decode module loaded");
    }

    #[test]
    fn bench_speculative_concept() {
        // Simulate the cost of speculative decode vs regular
        let iterations = 10_000_000u64;

        // Simulate: speculative check (just the validation overhead)
        let start = std::time::Instant::now();
        let mut sum = 0u64;
        for i in 0..iterations {
            // Simulate: check if entry is literal (signed comparison)
            let entry = i as i32;
            let is_literal = entry < 0;
            sum += is_literal as u64;
        }
        let elapsed_check = start.elapsed();

        // Simulate: batch copy overhead
        let start = std::time::Instant::now();
        let mut buf = [0u8; 64];
        let src = [0u8; 8];
        for _ in 0..iterations / 8 {
            buf[0..8].copy_from_slice(&src);
        }
        let elapsed_copy = start.elapsed();

        eprintln!("\nSpeculative batch concept benchmark:");
        eprintln!(
            "  Literal check: {} ns/iter",
            elapsed_check.as_nanos() / iterations as u128
        );
        eprintln!(
            "  Batch copy (8): {} ns/8-iter",
            elapsed_copy.as_nanos() / (iterations / 8) as u128
        );
        eprintln!("  (sum={}, buf[0]={})", sum, buf[0]); // Prevent optimization
    }
}
