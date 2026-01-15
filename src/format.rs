#![allow(dead_code)]

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    Gzip,
    Zlib,
    Zip,
}

impl CompressionFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "gz" | "gzip" => Some(CompressionFormat::Gzip),
            "zz" => Some(CompressionFormat::Zlib),
            "zip" => Some(CompressionFormat::Zip),
            _ => None,
        }
    }

    pub fn default_extension(&self) -> &'static str {
        match self {
            CompressionFormat::Gzip => ".gz",
            CompressionFormat::Zlib => ".zz",
            CompressionFormat::Zip => ".zip",
        }
    }

    pub fn magic_bytes(&self) -> &'static [u8] {
        match self {
            CompressionFormat::Gzip => &[0x1f, 0x8b],
            CompressionFormat::Zlib => &[0x78],
            CompressionFormat::Zip => &[0x50, 0x4b, 0x03, 0x04],
        }
    }
}

impl fmt::Display for CompressionFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionFormat::Gzip => write!(f, "gzip"),
            CompressionFormat::Zlib => write!(f, "zlib"),
            CompressionFormat::Zip => write!(f, "zip"),
        }
    }
}

impl Default for CompressionFormat {
    fn default() -> Self {
        CompressionFormat::Gzip
    }
}
