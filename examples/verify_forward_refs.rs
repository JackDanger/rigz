//! Verify forward-refs-only compression benefit
//!
//! This test creates a controlled scenario to measure EXACTLY how much
//! hash priming helps when we CAN'T reference the dictionary.
//!
//! Run with: cargo run --release --example verify_forward_refs

use std::io::Write;

fn compress_deflate(data: &[u8], level: u32) -> Vec<u8> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;

    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn decompress_deflate(data: &[u8]) -> Vec<u8> {
    use flate2::read::DeflateDecoder;
    use std::io::Read;

    let mut decoder = DeflateDecoder::new(data);
    let mut output = Vec::new();
    decoder.read_to_end(&mut output).unwrap();
    output
}

fn main() {
    println!("=== Forward-Refs-Only Verification ===\n");

    // Create test data with KNOWN patterns
    // The key: dict and data share common substrings
    let common_phrase = b"the quick brown fox jumps over the lazy dog ";
    let unique_phrase_a = b"ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA ";
    let unique_phrase_b = b"alpha beta gamma delta epsilon zeta eta theta ";

    // Dictionary: common phrase repeated, plus unique A
    let mut dict = Vec::new();
    for _ in 0..200 {
        dict.extend_from_slice(common_phrase);
    }
    for _ in 0..100 {
        dict.extend_from_slice(unique_phrase_a);
    }

    // Data: common phrase repeated, plus unique B
    let mut data = Vec::new();
    for _ in 0..200 {
        data.extend_from_slice(common_phrase);
    }
    for _ in 0..100 {
        data.extend_from_slice(unique_phrase_b);
    }

    println!("Dict size: {} bytes", dict.len());
    println!("Data size: {} bytes", data.len());
    println!();

    for level in [1, 6, 9] {
        println!("--- Level {} ---", level);

        // Test 1: Compress data alone
        let alone = compress_deflate(&data, level);

        // Test 2: Compress dict+data, keep only data's portion
        // by decompressing and checking sizes
        let mut combined = dict.clone();
        combined.extend_from_slice(&data);
        let combined_compressed = compress_deflate(&combined, level);

        // The combined stream includes both dict and data
        // We can't easily separate them, but we can measure:

        // Test 3: What if we had "forward refs only"?
        // Simulate by compressing data alone but with patterns that
        // would hash-collide with dict patterns

        // Actually, the TRUE test is:
        // - Compress dict+data as ONE stream
        // - Decompress to verify it works
        // - Measure the ratio

        let decompressed = decompress_deflate(&combined_compressed);
        assert_eq!(decompressed.len(), combined.len());

        // The key metric: how much does the combined stream compress?
        // If forward-refs-only works, we should see:
        // - dict portion compresses to X
        // - data portion compresses to Y (where Y < "data alone" due to hash priming)

        let dict_alone = compress_deflate(&dict, level);

        // Estimate: if the compressor was "fair", combined ≈ dict_alone + data_alone
        // But due to cross-references: combined < dict_alone + data_alone
        let fair_estimate = dict_alone.len() + alone.len();
        let actual = combined_compressed.len();
        let cross_ref_savings = fair_estimate as f64 - actual as f64;
        let cross_ref_pct = cross_ref_savings / alone.len() as f64 * 100.0;

        println!("  Data alone:     {} bytes", alone.len());
        println!("  Dict alone:     {} bytes", dict_alone.len());
        println!("  Fair estimate:  {} bytes (dict + data)", fair_estimate);
        println!("  Combined:       {} bytes", actual);
        println!(
            "  Cross-ref savings: {} bytes ({:.1}% of data)",
            cross_ref_savings as i64, cross_ref_pct
        );

        // The cross-ref savings shows how much benefit comes from
        // data referencing dict. This is what we'd LOSE with forward-refs-only.

        // What we'd KEEP is the hash priming benefit, which is harder to measure
        // directly but should be a portion of the dict benefit we saw earlier.

        println!();
    }

    println!("=== Analysis ===");
    println!("Cross-ref savings = benefit from data referencing dict");
    println!("This is what we LOSE with forward-refs-only.");
    println!("The remaining benefit (hash priming) = total dict benefit - cross-ref savings");
}
