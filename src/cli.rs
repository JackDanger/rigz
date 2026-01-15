use std::env;
// use std::path::PathBuf; // May be needed later
use crate::error::{RigzError, RigzResult};
use crate::format::CompressionFormat;

#[derive(Debug, Clone)]
pub struct RigzArgs {
    pub files: Vec<String>,
    pub compression_level: u8,
    pub block_size: usize,
    pub processes: usize,
    pub decompress: bool,
    pub test: bool,
    pub list: bool,
    pub stdout: bool,
    pub keep: bool,
    pub force: bool,
    pub quiet: bool,
    pub verbose: bool,
    pub verbosity: u8,
    pub recursive: bool,
    pub independent: bool,
    pub suffix: String,
    pub format: CompressionFormat,
    pub help: bool,
    pub version: bool,
    pub license: bool,
    pub rsyncable: bool,
    pub no_name: bool,
    pub no_time: bool,
    pub name: bool,
    pub time: bool,
    pub comment: Option<String>,
    pub alias: Option<String>,
    pub huffman: bool,
    pub rle: bool,
    pub synchronous: bool,
    pub best: bool,
    pub fast: bool,
}

impl Default for RigzArgs {
    fn default() -> Self {
        RigzArgs {
            files: Vec::new(),
            compression_level: 6,
            block_size: 128 * 1024, // 128KB
            processes: num_cpus::get().max(1),
            decompress: false,
            test: false,
            list: false,
            stdout: false,
            keep: false,
            force: false,
            quiet: false,
            verbose: false,
            verbosity: 1,
            recursive: false,
            independent: false,
            suffix: ".gz".to_string(),
            format: CompressionFormat::Gzip,
            help: false,
            version: false,
            license: false,
            rsyncable: false,
            no_name: false,
            no_time: false,
            name: true,
            time: true,
            comment: None,
            alias: None,
            huffman: false,
            rle: false,
            synchronous: false,
            best: false,
            fast: false,
        }
    }
}

impl RigzArgs {
    pub fn parse() -> RigzResult<Self> {
        let mut args = RigzArgs::default();
        let mut argv: Vec<String> = env::args().collect();
        argv.remove(0); // Remove program name

        // Check for environment variables first
        if let Ok(gzip_env) = env::var("GZIP") {
            let gzip_args = parse_env_args(&gzip_env);
            argv.splice(0..0, gzip_args);
        }

        if let Ok(pigz_env) = env::var("PIGZ") {
            let pigz_args = parse_env_args(&pigz_env);
            argv.splice(0..0, pigz_args);
        }

        let mut i = 0;
        let mut in_options = true;

        while i < argv.len() {
            let arg = &argv[i];

            if !in_options || !arg.starts_with('-') {
                args.files.push(arg.clone());
                i += 1;
                continue;
            }

            if arg == "--" {
                in_options = false;
                i += 1;
                continue;
            }

            if arg == "-" {
                args.files.push(arg.clone());
                i += 1;
                continue;
            }

            // Parse long options
            if arg.starts_with("--") {
                match arg.as_str() {
                    "--help" => args.help = true,
                    "--version" => args.version = true,
                    "--license" => args.license = true,
                    "--decompress" | "--uncompress" => args.decompress = true,
                    "--test" => args.test = true,
                    "--list" => args.list = true,
                    "--stdout" | "--to-stdout" => args.stdout = true,
                    "--keep" => args.keep = true,
                    "--force" => args.force = true,
                    "--quiet" | "--silent" => {
                        args.quiet = true;
                        args.verbosity = 0;
                    }
                    "--verbose" => {
                        args.verbose = true;
                        args.verbosity = 2;
                    }
                    "--recursive" => args.recursive = true,
                    "--independent" => args.independent = true,
                    "--zlib" => {
                        args.format = CompressionFormat::Zlib;
                        args.suffix = ".zz".to_string();
                    }
                    "--zip" => {
                        args.format = CompressionFormat::Zip;
                        args.suffix = ".zip".to_string();
                    }
                    "--rsyncable" => args.rsyncable = true,
                    "--no-name" => {
                        args.no_name = true;
                        args.name = false;
                    }
                    "--no-time" => {
                        args.no_time = true;
                        args.time = false;
                    }
                    "--name" => {
                        args.name = true;
                        args.no_name = false;
                    }
                    "--time" => {
                        args.time = true;
                        args.no_time = false;
                    }
                    "--huffman" => args.huffman = true,
                    "--rle" => args.rle = true,
                    "--synchronous" => args.synchronous = true,
                    "--best" => {
                        args.best = true;
                        args.compression_level = 9;
                    }
                    "--fast" => {
                        args.fast = true;
                        args.compression_level = 1;
                    }
                    _ => {
                        // Handle options with values
                        if arg.starts_with("--blocksize=") {
                            let value = &arg[12..];
                            args.block_size = parse_block_size(value)?;
                        } else if arg.starts_with("--processes=") {
                            let value = &arg[12..];
                            args.processes = value.parse().map_err(|_| {
                                RigzError::invalid_argument(format!("Invalid processes: {}", value))
                            })?;
                        } else if arg.starts_with("--suffix=") {
                            let value = &arg[9..];
                            args.suffix = value.to_string();
                        } else if arg.starts_with("--comment=") {
                            let value = &arg[10..];
                            args.comment = Some(value.to_string());
                        } else if arg.starts_with("--alias=") {
                            let value = &arg[8..];
                            args.alias = Some(value.to_string());
                        } else if arg == "--blocksize"
                            || arg == "--processes"
                            || arg == "--suffix"
                            || arg == "--comment"
                            || arg == "--alias"
                        {
                            // These require the next argument
                            if i + 1 >= argv.len() {
                                return Err(RigzError::invalid_argument(format!(
                                    "{} requires an argument",
                                    arg
                                )));
                            }
                            i += 1;
                            let value = &argv[i];

                            match arg.as_str() {
                                "--blocksize" => args.block_size = parse_block_size(value)?,
                                "--processes" => {
                                    args.processes = value.parse().map_err(|_| {
                                        RigzError::invalid_argument(format!(
                                            "Invalid processes: {}",
                                            value
                                        ))
                                    })?
                                }
                                "--suffix" => args.suffix = value.to_string(),
                                "--comment" => args.comment = Some(value.to_string()),
                                "--alias" => args.alias = Some(value.to_string()),
                                _ => unreachable!(),
                            }
                        } else {
                            return Err(RigzError::invalid_argument(format!(
                                "Unknown option: {}",
                                arg
                            )));
                        }
                    }
                }
            } else {
                // Parse short options
                let chars: Vec<char> = arg.chars().collect();
                let mut j = 1; // Skip the initial '-'

                while j < chars.len() {
                    match chars[j] {
                        'h' => args.help = true,
                        'V' => args.version = true,
                        'L' => args.license = true,
                        'd' => args.decompress = true,
                        't' => args.test = true,
                        'l' => args.list = true,
                        'c' => args.stdout = true,
                        'k' => args.keep = true,
                        'f' => args.force = true,
                        'q' => {
                            args.quiet = true;
                            args.verbosity = 0;
                        }
                        'v' => {
                            args.verbose = true;
                            args.verbosity += 1;
                        }
                        'r' => args.recursive = true,
                        'i' => args.independent = true,
                        'z' => {
                            args.format = CompressionFormat::Zlib;
                            args.suffix = ".zz".to_string();
                        }
                        'K' => {
                            args.format = CompressionFormat::Zip;
                            args.suffix = ".zip".to_string();
                        }
                        'R' => args.rsyncable = true,
                        'n' => {
                            args.no_name = true;
                            args.name = false;
                        }
                        'N' => {
                            args.name = true;
                            args.no_name = false;
                        }
                        'H' => args.huffman = true,
                        'U' => args.rle = true,
                        'Y' => args.synchronous = true,
                        '0'..='9' => {
                            let level = chars[j] as u8 - b'0';
                            args.compression_level = level;
                        }
                        'b' | 'p' | 'S' | 'C' | 'A' => {
                            // Save the option character before potentially modifying j
                            let opt_char = chars[j];
                            
                            // These require values
                            let value = if j + 1 < chars.len() {
                                // Value is attached to the option
                                let value_str: String = chars[j + 1..].iter().collect();
                                j = chars.len(); // Skip the rest of the characters
                                value_str
                            } else {
                                // Value is the next argument
                                if i + 1 >= argv.len() {
                                    return Err(RigzError::invalid_argument(format!(
                                        "-{} requires an argument",
                                        opt_char
                                    )));
                                }
                                i += 1;
                                argv[i].clone()
                            };

                            match opt_char {
                                'b' => args.block_size = parse_block_size(&value)?,
                                'p' => {
                                    args.processes = value.parse().map_err(|_| {
                                        RigzError::invalid_argument(format!(
                                            "Invalid processes: {}",
                                            value
                                        ))
                                    })?
                                }
                                'S' => args.suffix = value,
                                'C' => args.comment = Some(value),
                                'A' => args.alias = Some(value),
                                _ => unreachable!(),
                            }
                        }
                        _ => {
                            return Err(RigzError::invalid_argument(format!(
                                "Unknown option: -{}",
                                chars[j]
                            )))
                        }
                    }
                    j += 1;
                }
            }

            i += 1;
        }

        // Validate and adjust settings
        if args.processes == 0 {
            args.processes = 1;
        }

        if args.compression_level > 9 {
            return Err(RigzError::InvalidLevel(args.compression_level));
        }

        if args.block_size < 1024 {
            return Err(RigzError::InvalidBlockSize(
                "Block size must be at least 1K".to_string(),
            ));
        }

        Ok(args)
    }
}

fn parse_env_args(env_str: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current_arg = String::new();
    let mut in_quotes = false;
    let mut chars = env_str.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ' ' | '\t' if !in_quotes => {
                if !current_arg.is_empty() {
                    args.push(current_arg.clone());
                    current_arg.clear();
                }
            }
            _ => current_arg.push(ch),
        }
    }

    if !current_arg.is_empty() {
        args.push(current_arg);
    }

    args
}

fn parse_block_size(value: &str) -> RigzResult<usize> {
    let value = value.to_lowercase();

    if value.is_empty() {
        return Err(RigzError::InvalidBlockSize("Empty block size".to_string()));
    }

    let (num_str, multiplier) = if value.ends_with('k') {
        (&value[..value.len() - 1], 1024)
    } else if value.ends_with('m') {
        (&value[..value.len() - 1], 1024 * 1024)
    } else if value.ends_with('g') {
        (&value[..value.len() - 1], 1024 * 1024 * 1024)
    } else {
        (value.as_str(), 1)
    };

    let num: usize = num_str
        .parse()
        .map_err(|_| RigzError::InvalidBlockSize(format!("Invalid block size: {}", value)))?;

    Ok(num * multiplier)
}
