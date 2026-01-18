//! Test the benefit of hash priming without dictionary references
//!
//! This experiment measures if hash table priming alone (without dictionary refs)
//! provides any compression benefit.
//!
//! Run with: cargo run --release --example test_hash_priming

use std::io::Write;

fn compress_no_dict(data: &[u8], level: u32) -> Vec<u8> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;

    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn compress_with_dict(data: &[u8], dict: &[u8], level: u32) -> Vec<u8> {
    use flate2::{Compress, Compression, FlushCompress};

    let mut compress = Compress::new(Compression::new(level), false);

    // Set dictionary (primes hash tables AND sliding window)
    let dict_slice = if dict.len() > 32768 {
        &dict[dict.len() - 32768..]
    } else {
        dict
    };
    compress.set_dictionary(dict_slice).unwrap();

    // Compress
    let mut output = vec![0u8; data.len() + 1024];
    compress
        .compress(data, &mut output, FlushCompress::Finish)
        .unwrap();
    output.truncate(compress.total_out() as usize);
    output
}

/// Simulate "forward refs only" by compressing overlapping data
/// and measuring how much comes from the overlap vs the unique part
fn estimate_forward_refs_benefit(data: &[u8], dict: &[u8], level: u32) -> (usize, usize, f64) {
    // Method 1: Compress data alone (no dict)
    let no_dict_size = compress_no_dict(data, level).len();

    // Method 2: Compress (dict + data) as one unit, measure total
    // This simulates what "forward refs only" would produce:
    // - Hash tables primed with dict
    // - But references stay within data portion
    let mut combined = dict.to_vec();
    combined.extend_from_slice(data);
    let combined_size = compress_no_dict(&combined, level).len();

    // The difference tells us how much the hash priming helps
    // (combined_size includes dict, so subtract expected dict contribution)
    let dict_alone_size = compress_no_dict(dict, level).len();
    let estimated_data_portion = combined_size.saturating_sub(dict_alone_size);

    let benefit = if no_dict_size > 0 {
        (no_dict_size as f64 - estimated_data_portion as f64) / no_dict_size as f64 * 100.0
    } else {
        0.0
    };

    (no_dict_size, estimated_data_portion, benefit)
}

fn main() {
    println!("=== Hash Priming Benefit Analysis ===\n");

    // Load test data
    let seed = if std::path::Path::new("test_data/text-1MB.txt").exists() {
        std::fs::read("test_data/text-1MB.txt").unwrap()
    } else {
        b"The quick brown fox jumps over the lazy dog. ".repeat(20000)
    };

    let block_size = 64 * 1024; // 64KB
    let dict_size = 32 * 1024; // 32KB

    println!(
        "Block size: {}KB, Dictionary: {}KB\n",
        block_size / 1024,
        dict_size / 1024
    );

    for level in [1, 6, 9] {
        println!("--- Level {} ---", level);

        // Simulate multiple blocks with overlapping dictionaries
        let mut total_no_dict = 0;
        let mut total_with_dict = 0;
        let mut total_estimated_forward = 0;
        let num_blocks = 10;

        for i in 0..num_blocks {
            let start: usize = i * block_size;
            let dict_start: usize = start.saturating_sub(dict_size);

            if start + block_size > seed.len() {
                break;
            }

            let dict = &seed[dict_start..start];
            let data = &seed[start..start + block_size];

            let no_dict = compress_no_dict(data, level).len();
            let with_dict = compress_with_dict(data, dict, level).len();
            let (_, estimated_forward, _) = estimate_forward_refs_benefit(data, dict, level);

            total_no_dict += no_dict;
            total_with_dict += with_dict;
            total_estimated_forward += estimated_forward;
        }

        let dict_benefit =
            (total_no_dict as f64 - total_with_dict as f64) / total_no_dict as f64 * 100.0;
        let forward_benefit =
            (total_no_dict as f64 - total_estimated_forward as f64) / total_no_dict as f64 * 100.0;

        println!("  No dictionary:    {} bytes", total_no_dict);
        println!(
            "  With dictionary:  {} bytes ({:+.1}%)",
            total_with_dict, -dict_benefit
        );
        println!(
            "  Forward-refs est: {} bytes ({:+.1}%)",
            total_estimated_forward, -forward_benefit
        );
        println!(
            "  Hash priming benefit: {:.1}% (of {:.1}% total dict benefit)",
            forward_benefit, dict_benefit
        );
        println!();
    }

    println!("=== Interpretation ===");
    println!("'Forward-refs est' approximates what we'd get with hash priming");
    println!("but no dictionary references. If this is close to 'No dictionary',");
    println!("then the forward-refs-only approach provides minimal benefit.");
}
