#![allow(dead_code)]

use std::fs;
use std::path::Path;

use crate::error::{RigzError, RigzResult};

pub fn get_file_metadata(path: &Path) -> RigzResult<fs::Metadata> {
    fs::metadata(path).map_err(RigzError::Io)
}

pub fn detect_format_from_file(path: &Path) -> Option<crate::format::CompressionFormat> {
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return crate::format::CompressionFormat::from_extension(ext_str);
        }
    }
    None
}

pub fn is_compressed_file(path: &Path) -> bool {
    detect_format_from_file(path).is_some()
}

pub fn strip_compression_extension(path: &Path) -> std::path::PathBuf {
    let mut result = path.to_path_buf();

    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            if crate::format::CompressionFormat::from_extension(ext_str).is_some() {
                result.set_extension("");
            }
        }
    }

    result
}

pub fn format_size(size: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = size as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{:.0} {}", size, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

pub fn format_percentage(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "N/A".to_string()
    } else {
        let percentage = (numerator as f64 / denominator as f64) * 100.0;
        format!("{:.1}%", percentage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(50, 100), "50.0%");
        assert_eq!(format_percentage(0, 100), "0.0%");
        assert_eq!(format_percentage(100, 0), "N/A");
    }
}
