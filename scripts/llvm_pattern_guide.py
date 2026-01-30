#!/usr/bin/env python3
"""
LLVM Pattern Guide - Help LLVM Generate Optimal Code

Instead of fighting LLVM with inline ASM, this script analyzes
LLVM's output and suggests Rust code patterns that lead to
optimal instruction generation.

The insight: LLVM is smarter than us at scheduling and register allocation.
We should write code that LLVM can optimize well, not override it.
"""

import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

WORKSPACE = Path(__file__).parent.parent

# ============================================================================
# Rust Patterns that Generate Optimal ARM64 Code
# ============================================================================

OPTIMAL_PATTERNS = {
    # Pattern: BFXIL for updating low bits while preserving high bits
    "bfxil": {
        "description": "BFXIL: Bit field extract and insert low",
        "bad_rust": """
// This generates AND + ORR
let result = (dest & !mask) | (src & mask);
""",
        "good_rust": """
// This often generates BFXIL on ARM64
let preserved_high = dest & !((1u32 << width) - 1);
let new_low = src & ((1u32 << width) - 1);
let result = preserved_high | new_low;

// Even better - let LLVM see the pattern clearly:
bitsleft = (bits_u8 as u32) | 56;  // LLVM recognizes "insert 56 into high bits"
""",
        "check": lambda asm: "bfxil" in asm.lower(),
    },
    
    # Pattern: CCMP for chained conditions
    "ccmp": {
        "description": "CCMP: Conditional compare for chained conditions",
        "bad_rust": """
// This generates two separate CMP + branch sequences
if in_pos >= in_end || out_pos >= out_end {
    return slowpath();
}
""",
        "good_rust": """
// Structure conditions to enable CCMP:
// LLVM will generate: cmp; ccmp (if first cmp was true)
let in_ok = in_pos < in_end;
let out_ok = out_pos < out_end;
if !in_ok || !out_ok {
    return slowpath();
}

// Or use && which LLVM handles well:
if in_pos < in_end && out_pos < out_end {
    // fastpath
}
""",
        "check": lambda asm: "ccmp" in asm.lower(),
    },
    
    # Pattern: UBFX for bit extraction
    "ubfx": {
        "description": "UBFX: Unsigned bit field extract",
        "bad_rust": """
// This generates LSR + AND
let bits = (value >> shift) & mask;
""",
        "good_rust": """
// LLVM generates UBFX when it sees constant bit range:
let bits = (entry >> 16) & 0x1ff;  // 9-bit field at position 16

// Even better - use u8/u16 casts for clarity:
let low_byte = entry as u8;  // LLVM knows this is bits 0-7
""",
        "check": lambda asm: "ubfx" in asm.lower(),
    },
    
    # Pattern: TBZ/TBNZ for single-bit tests
    "tbz": {
        "description": "TBZ/TBNZ: Test bit and branch",
        "bad_rust": """
// This generates AND + CMP + branch
if (value & (1 << bit)) != 0 {
    ...
}
""",
        "good_rust": """
// LLVM generates TBNZ for this pattern:
if entry & (1 << 31) != 0 {  // Test bit 31
    // literal path
}

// Or use explicit bit test:
if entry.wrapping_shr(31) & 1 != 0 {
    // literal path
}
""",
        "check": lambda asm: "tbz" in asm.lower() or "tbnz" in asm.lower(),
    },
    
    # Pattern: CSEL for branchless conditionals
    "csel": {
        "description": "CSEL: Conditional select",
        "bad_rust": """
// Branches can hurt performance in tight loops
let result = if condition { a } else { b };
""",
        "good_rust": """
// LLVM often generates CSEL for simple conditions:
let result = if cond { a } else { b };

// Ensure both arms are simple (no side effects):
// LLVM is more likely to use CSEL

// Even better - use built-in conditional methods:
let result = cond.then_some(a).unwrap_or(b);  // Not always better
""",
        "check": lambda asm: "csel" in asm.lower(),
    },
    
    # Pattern: LDP/STP for paired loads/stores
    "ldp_stp": {
        "description": "LDP/STP: Load/store pair for adjacent memory",
        "bad_rust": """
// Two separate loads
let a = ptr.add(0).read();
let b = ptr.add(8).read();
""",
        "good_rust": """
// LLVM generates LDP when it sees adjacent accesses:
let a = ptr.add(0).read();
let b = ptr.add(8).read();  // LLVM will combine into LDP

// For writes, same principle:
ptr.add(0).write(a);
ptr.add(8).write(b);  // LLVM combines into STP

// CRITICAL: Keep accesses adjacent in source code!
// Don't interleave with other operations
""",
        "check": lambda asm: "ldp" in asm.lower() or "stp" in asm.lower(),
    },
}

# ============================================================================
# ASM Analyzer
# ============================================================================

@dataclass
class PatternAnalysis:
    """Analysis of patterns in generated ASM."""
    pattern_name: str
    found: bool
    count: int
    locations: List[str]
    suggestion: str


def analyze_asm_for_patterns(asm_text: str) -> List[PatternAnalysis]:
    """Analyze ASM text for optimal patterns."""
    results = []
    
    for name, pattern in OPTIMAL_PATTERNS.items():
        check_fn = pattern.get("check", lambda x: False)
        found = check_fn(asm_text)
        
        # Count occurrences
        count = asm_text.lower().count(name.split("_")[0])  # Handle ldp_stp -> ldp
        
        # Find example locations
        locations = []
        for i, line in enumerate(asm_text.split('\n')):
            if name.split("_")[0] in line.lower():
                locations.append(f"Line {i}: {line.strip()[:60]}")
                if len(locations) >= 3:
                    break
        
        if found:
            suggestion = f"✓ {pattern['description']} is being generated ({count} occurrences)"
        else:
            suggestion = f"✗ {pattern['description']} not found - consider restructuring code"
        
        results.append(PatternAnalysis(
            pattern_name=name,
            found=found,
            count=count,
            locations=locations,
            suggestion=suggestion,
        ))
    
    return results


def analyze_instruction_mix(asm_text: str) -> Dict[str, int]:
    """Count instruction types in ASM."""
    counts = {}
    
    for line in asm_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('.') or line.startswith('//') or ':' in line:
            continue
        
        parts = line.split()
        if parts:
            opcode = parts[0].lower()
            counts[opcode] = counts.get(opcode, 0) + 1
    
    return counts


def compare_rust_vs_c(rust_asm: str, c_asm: str) -> List[str]:
    """Compare instruction mix between Rust and C generated code."""
    rust_mix = analyze_instruction_mix(rust_asm)
    c_mix = analyze_instruction_mix(c_asm)
    
    differences = []
    
    all_opcodes = set(rust_mix.keys()) | set(c_mix.keys())
    
    for opcode in sorted(all_opcodes):
        rust_count = rust_mix.get(opcode, 0)
        c_count = c_mix.get(opcode, 0)
        
        if rust_count != c_count:
            delta = rust_count - c_count
            if abs(delta) >= 2:  # Significant difference
                if delta > 0:
                    differences.append(f"  {opcode}: Rust has {delta} MORE ({rust_count} vs {c_count})")
                else:
                    differences.append(f"  {opcode}: C has {-delta} MORE ({c_count} vs {rust_count})")
    
    return differences


# ============================================================================
# Code Suggestions
# ============================================================================

def generate_suggestions(analysis: List[PatternAnalysis]) -> str:
    """Generate code suggestions based on analysis."""
    
    suggestions = ["# Suggestions for Better LLVM Codegen\n"]
    
    for result in analysis:
        if not result.found:
            pattern = OPTIMAL_PATTERNS[result.pattern_name]
            suggestions.append(f"\n## {result.pattern_name.upper()}: {pattern['description']}\n")
            suggestions.append(f"**Current (suboptimal):**\n```rust{pattern['bad_rust']}```\n")
            suggestions.append(f"**Suggested (optimal):**\n```rust{pattern['good_rust']}```\n")
    
    return "\n".join(suggestions)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLVM Pattern Guide')
    parser.add_argument('--analyze', action='store_true', help='Analyze current ASM')
    parser.add_argument('--compare', action='store_true', help='Compare Rust vs C')
    parser.add_argument('--suggest', action='store_true', help='Generate suggestions')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLVM PATTERN GUIDE - Help LLVM Generate Optimal Code")
    print("=" * 70)
    
    # Load Rust ASM
    asm_files = list((WORKSPACE / "target/release/deps").glob("gzippy*.s"))
    if not asm_files:
        print("\nGenerating Rust ASM...")
        subprocess.run(
            ['cargo', 'rustc', '--release', '--bin', 'gzippy', '--', '--emit=asm'],
            cwd=WORKSPACE,
            capture_output=True
        )
        asm_files = list((WORKSPACE / "target/release/deps").glob("gzippy*.s"))
    
    if asm_files:
        # Use newest
        asm_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        rust_asm = asm_files[0].read_text()
        
        # Find decode function
        if 'decode_huffman_libdeflate_style' in rust_asm:
            start = rust_asm.find('decode_huffman_libdeflate_style')
            end = rust_asm.find('\n.cfi_endproc', start)
            if end > start:
                decode_asm = rust_asm[start:end]
            else:
                decode_asm = rust_asm[start:start+5000]
        else:
            decode_asm = rust_asm[:10000]
        
        print(f"\nLoaded Rust ASM: {len(rust_asm)} bytes")
        print(f"Decode function: {len(decode_asm)} bytes")
    else:
        decode_asm = ""
        print("\nNo Rust ASM found")
    
    # Analyze patterns
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)
    
    if decode_asm:
        analysis = analyze_asm_for_patterns(decode_asm)
        
        for result in analysis:
            status = "✓" if result.found else "✗"
            print(f"\n{status} {result.pattern_name.upper()}: {result.suggestion}")
            if result.found and result.locations:
                for loc in result.locations[:2]:
                    print(f"    {loc}")
    
    # Compare with C
    c_asm_file = WORKSPACE / "target" / "libdeflate_decompress.s"
    if c_asm_file.exists() and args.compare:
        print("\n" + "=" * 70)
        print("RUST VS C COMPARISON")
        print("=" * 70)
        
        c_asm = c_asm_file.read_text()
        differences = compare_rust_vs_c(decode_asm, c_asm)
        
        if differences:
            print("\nSignificant instruction count differences:")
            for diff in differences[:15]:
                print(diff)
        else:
            print("\nInstruction mix is similar!")
    
    # Suggestions
    if args.suggest:
        print("\n" + "=" * 70)
        print("CODE SUGGESTIONS")
        print("=" * 70)
        
        if decode_asm:
            analysis = analyze_asm_for_patterns(decode_asm)
            suggestions = generate_suggestions(analysis)
            print(suggestions)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Insight: Don't fight LLVM with inline ASM!

Instead:
1. Write clean Rust code that LLVM can optimize
2. Use patterns that lead to optimal instruction selection
3. Trust LLVM's register allocation and scheduling

The ~6% gap between Rust and C is due to:
- Different LLVM versions (Rust uses older LLVM)
- Different default optimization flags
- Some Rust-specific safety checks

To improve:
1. Use `-C target-cpu=native` for better codegen
2. Structure code to enable CCMP, UBFX, BFXIL patterns
3. Keep hot paths simple and predictable
""")


if __name__ == '__main__':
    main()
