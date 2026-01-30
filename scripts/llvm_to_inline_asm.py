#!/usr/bin/env python3
"""
LLVM to Inline ASM Converter v3

This script extracts LLVM's generated assembly for the decode function
and converts it into Rust inline ASM format that can be directly used
in our implementation.

Key features:
1. Extracts only the hot loop (skips error handling)
2. Creates proper exit points for slow paths
3. Maps registers to inline ASM placeholders
4. Handles forward/backward label references correctly

Usage:
    python3 scripts/llvm_to_inline_asm.py

Output:
    - target/llvm_generated_full.rs - Ready-to-use inline ASM
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

@dataclass
class AsmInstruction:
    """Parsed assembly instruction"""
    label: Optional[str] = None
    opcode: str = ""
    operands: List[str] = field(default_factory=list)
    raw: str = ""
    is_directive: bool = False
    index: int = 0
    
    def is_external_call(self) -> bool:
        """Check if this is a call to an external Rust function"""
        if self.opcode in ['bl', 'b']:
            for op in self.operands:
                if '__ZN' in op or '_$' in op:
                    return True
        return False
    
    def is_epilogue(self) -> bool:
        """Check if this is part of the function epilogue"""
        return self.opcode in ['ret'] or (self.opcode == 'ldp' and any('x29' in op or 'x30' in op for op in self.operands))
    
    def to_inline_asm(self, reg_map: Dict[str, str], label_map: Dict[str, str], 
                      label_positions: Dict[str, int], exit_label: str = "99") -> Optional[str]:
        """Convert to Rust inline ASM format"""
        if self.is_directive:
            return None
        
        if self.label:
            mapped_label = label_map.get(self.label, self.label)
            return f'        "{mapped_label}:",'
        
        # Map operands
        mapped_operands = []
        for op in self.operands:
            mapped = self._map_operand(op, reg_map, label_map, label_positions, exit_label)
            mapped_operands.append(mapped)
        
        operands_str = ", ".join(mapped_operands)
        if operands_str:
            return f'        "{self.opcode} {operands_str}",'
        return f'        "{self.opcode}",'
    
    def _map_operand(self, operand: str, reg_map: Dict[str, str], label_map: Dict[str, str],
                     label_positions: Dict[str, int], exit_label: str) -> str:
        result = operand
        
        # Map labels with correct direction
        for llvm_label, num_label in label_map.items():
            target_pos = label_positions.get(llvm_label, float('inf'))
            direction = 'f' if target_pos > self.index else 'b'
            result = re.sub(rf'\b{re.escape(llvm_label)}\b', num_label + direction, result)
        
        # Map registers
        for llvm_reg, placeholder in sorted(reg_map.items(), key=lambda x: -len(x[0])):
            if llvm_reg.startswith('x'):
                base = llvm_reg[1:]
                w_reg = 'w' + base
                result = re.sub(rf'\b{llvm_reg}\b', placeholder, result)
                w_placeholder = placeholder.replace('}', ':w}')
                result = re.sub(rf'\b{w_reg}\b', w_placeholder, result)
        
        return result

def find_asm_file() -> Optional[Path]:
    """Find the generated .s file"""
    deps_dir = Path("target/release/deps")
    if not deps_dir.exists():
        return None
    
    for f in sorted(deps_dir.glob("gzippy-*.s"), key=lambda x: x.stat().st_size, reverse=True):
        if f.stat().st_size > 1_000_000:
            return f
    return None

def parse_instruction(line: str) -> Optional[AsmInstruction]:
    """Parse a single assembly line"""
    original = line
    line = line.strip()
    
    if not line:
        return None
    
    if line.startswith('.') or line.startswith(';') or line.startswith('//'):
        return AsmInstruction(is_directive=True, raw=line)
    
    # Label
    if line.endswith(':'):
        label = line[:-1]
        if label.startswith('Lfunc') or label.startswith('Ltmp') or label.startswith('Lloh'):
            return AsmInstruction(is_directive=True, raw=line)
        return AsmInstruction(label=label, raw=original)
    
    # Parse instruction
    parts = line.split(None, 1)
    opcode = parts[0].lower()
    
    operands = []
    if len(parts) > 1:
        operand_str = parts[1].split('//')[0].strip()  # Remove comments
        current = ""
        bracket_depth = 0
        for char in operand_str:
            if char == '[':
                bracket_depth += 1
                current += char
            elif char == ']':
                bracket_depth -= 1
                current += char
            elif char == ',' and bracket_depth == 0:
                if current.strip():
                    operands.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            operands.append(current.strip())
    
    return AsmInstruction(opcode=opcode, operands=operands, raw=original)

def find_function_bounds(asm_content: str, func_pattern: str) -> Tuple[int, int]:
    """Find function line numbers"""
    lines = asm_content.split('\n')
    
    start = None
    for i, line in enumerate(lines, 1):
        if func_pattern in line and line.strip().endswith(':'):
            start = i
            break
    
    if not start:
        return 0, 0
    
    end = len(lines)
    for i, line in enumerate(lines[start:], start + 1):
        stripped = line.strip()
        if (stripped.startswith('__ZN') or stripped.startswith('_')) and stripped.endswith(':'):
            if not stripped.startswith('__ZN6gzippy20consume_first_decode31decode_huffman'):
                end = i
                break
    
    return start, end

def analyze_function(lines: List[str]) -> Tuple[List[AsmInstruction], Dict[str, int], Set[str]]:
    """Analyze function and return instructions, labels, and used registers"""
    instructions = []
    labels = {}
    used_regs = set()
    
    idx = 0
    for line in lines:
        inst = parse_instruction(line)
        if inst and not inst.is_directive:
            inst.index = idx
            instructions.append(inst)
            
            if inst.label:
                labels[inst.label] = idx
            
            for op in inst.operands:
                for m in re.findall(r'\b([xwq]\d+)\b', op):
                    used_regs.add(m)
            
            idx += 1
    
    return instructions, labels, used_regs

def find_hot_loop_range(instructions: List[AsmInstruction], labels: Dict[str, int]) -> Tuple[int, int]:
    """Find the hot loop start and end indices"""
    # Hot loop typically starts at LBB*_6 or LBB*_7 (after initial setup)
    # and ends before error handling (LBB*_125+)
    
    start_idx = 0
    end_idx = len(instructions)
    
    for inst in instructions:
        if inst.label:
            # Find first LBB*_6 or LBB*_7
            if re.match(r'LBB\d+_[67]$', inst.label):
                start_idx = inst.index
                break
    
    for inst in instructions:
        if inst.label:
            # Find first exit block (error handling)
            if re.match(r'LBB\d+_12[0-9]$', inst.label):
                end_idx = inst.index
                break
    
    return start_idx, end_idx

def create_label_map(labels: Dict[str, int]) -> Dict[str, str]:
    """Map LLVM labels to numeric labels"""
    label_map = {}
    for i, label in enumerate(sorted(labels.keys(), key=lambda l: labels[l])):
        if label.startswith('LBB'):
            label_map[label] = str(i + 1)
    return label_map

def create_register_map() -> Dict[str, str]:
    """Create register mapping - only core state variables"""
    return {
        'x10': '{in_pos}',
        'x11': '{bitbuf}',
        'x21': '{bitsleft}',
        'x3': '{out_pos}',
        'x22': '{entry}',
        'x8': '{in_ptr}',
        'x1': '{out_ptr}',
        'x2': '{out_len}',
        'x4': '{litlen_ptr}',
        'x6': '{dist_ptr}',
        'x9': '{in_end}',
    }

def generate_inline_asm(instructions: List[AsmInstruction], labels: Dict[str, int], 
                        hot_start: int, hot_end: int) -> str:
    """Generate the inline ASM code"""
    
    reg_map = create_register_map()
    label_map = create_label_map(labels)
    
    lines = []
    lines.append('// =============================================================================')
    lines.append('// AUTO-GENERATED FROM LLVM ASSEMBLY - Hot Loop Only')
    lines.append('// Regenerate with: python3 scripts/llvm_to_inline_asm.py')
    lines.append('// =============================================================================')
    lines.append('')
    lines.append('/// LLVM-generated decode hot loop')
    lines.append('/// This is an exact copy of LLVM\'s generated assembly')
    lines.append('#[cfg(target_arch = "aarch64")]')
    lines.append('#[inline(never)]')
    lines.append('pub unsafe fn decode_huffman_llvm_hotloop(')
    lines.append('    bits: &mut Bits,')
    lines.append('    output: &mut [u8],')
    lines.append('    mut out_pos: usize,')
    lines.append('    litlen: &LitLenTable,')
    lines.append('    dist: &DistTable,')
    lines.append(') -> Result<usize> {')
    lines.append('    use std::arch::asm;')
    lines.append('    ')
    lines.append('    // Setup from Rust')
    lines.append('    let out_ptr = output.as_mut_ptr();')
    lines.append('    let out_len = output.len();')
    lines.append('    let litlen_ptr = litlen.entries.as_ptr();')
    lines.append('    let dist_ptr = dist.entries.as_ptr();')
    lines.append('    ')
    lines.append('    let mut bitbuf: u64 = bits.bitbuf;')
    lines.append('    let mut bitsleft: u64 = bits.bitsleft as u64;')
    lines.append('    let mut in_pos: usize = bits.pos;')
    lines.append('    let in_ptr = bits.data.as_ptr();')
    lines.append('    let in_end: usize = bits.data.len().saturating_sub(32);')
    lines.append('    ')
    lines.append('    let mut entry: u64 = 0;')
    lines.append('    let mut exit_reason: u64 = 0;  // 0 = continue, 1 = EOB, 2 = error')
    lines.append('    ')
    lines.append('    // Initial entry lookup')
    lines.append('    entry = (*litlen_ptr.add(bitbuf as usize & 0x7ff)) as u64;')
    lines.append('    ')
    lines.append('    asm!(')
    lines.append('        // Jump to hot loop entry')
    lines.append('        "b 2f",')
    lines.append('        ')
    
    # Track which labels we actually use
    used_labels = set()
    
    # First pass: collect used labels
    for inst in instructions[hot_start:hot_end]:
        if not inst.is_directive and not inst.label:
            for op in inst.operands:
                for label in labels.keys():
                    if label in op:
                        used_labels.add(label)
    
    # Generate hot loop instructions
    lines.append('        // === HOT LOOP START ===')
    
    skip_until_label = False
    for inst in instructions[hot_start:hot_end]:
        # Skip external calls and error handling
        if inst.is_external_call():
            skip_until_label = True
            continue
        
        if inst.is_epilogue():
            continue
            
        if skip_until_label and inst.label:
            skip_until_label = False
        
        if skip_until_label:
            continue
        
        # Convert the instruction
        asm_line = inst.to_inline_asm(reg_map, label_map, labels)
        if asm_line:
            lines.append(asm_line)
    
    # Add exit point
    lines.append('        ')
    lines.append('        // === EXIT POINTS ===')
    lines.append('        "99:",  // Normal exit')
    lines.append('        "mov {exit_reason}, #0",')
    lines.append('        "b 100f",')
    lines.append('        ')
    lines.append('        "98:",  // EOB exit')
    lines.append('        "mov {exit_reason}, #1",')
    lines.append('        "b 100f",')
    lines.append('        ')
    lines.append('        "97:",  // Error exit')
    lines.append('        "mov {exit_reason}, #2",')
    lines.append('        ')
    lines.append('        "100:",  // Final exit')
    lines.append('        ')
    
    # Register constraints
    lines.append('        // === REGISTER BINDINGS ===')
    lines.append('        bitbuf = inout("x11") bitbuf,')
    lines.append('        bitsleft = inout("x21") bitsleft,')
    lines.append('        in_pos = inout("x10") in_pos,')
    lines.append('        out_pos = inout("x3") out_pos,')
    lines.append('        entry = inout("x22") entry,')
    lines.append('        exit_reason = out("x0") exit_reason,')
    lines.append('        ')
    lines.append('        in_ptr = in("x8") in_ptr,')
    lines.append('        out_ptr = in("x1") out_ptr,')
    lines.append('        out_len = in("x2") out_len,')
    lines.append('        litlen_ptr = in("x4") litlen_ptr,')
    lines.append('        dist_ptr = in("x6") dist_ptr,')
    lines.append('        in_end = in("x9") in_end,')
    lines.append('        ')
    
    # Clobbers
    scratch = ['x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x19', 'x20', 
               'x23', 'x24', 'x25', 'x26', 'x27', 'x28']
    for reg in scratch:
        lines.append(f'        out("{reg}") _,')
    
    for i in range(4):
        lines.append(f'        out("v{i}") _,')
    
    lines.append('        ')
    lines.append('        options(nostack),')
    lines.append('    );')
    lines.append('    ')
    lines.append('    // Sync state')
    lines.append('    bits.bitbuf = bitbuf;')
    lines.append('    bits.bitsleft = bitsleft as u32;')
    lines.append('    bits.pos = in_pos;')
    lines.append('    ')
    lines.append('    match exit_reason {')
    lines.append('        0 => Ok(out_pos),  // Normal')
    lines.append('        1 => Ok(out_pos),  // EOB')
    lines.append('        _ => Err(crate::Error::InvalidData),  // Error')
    lines.append('    }')
    lines.append('}')
    
    return '\n'.join(lines)

def main():
    print("=" * 70)
    print("LLVM to Inline ASM Converter v3")
    print("=" * 70)
    
    asm_file = find_asm_file()
    if not asm_file:
        print("ERROR: No assembly file found. Run:")
        print("  RUSTFLAGS='--emit asm' cargo build --release")
        sys.exit(1)
    
    print(f"Reading: {asm_file}")
    asm_content = asm_file.read_text()
    
    func_pattern = "decode_huffman_libdeflate_style"
    start, end = find_function_bounds(asm_content, func_pattern)
    
    if start == 0:
        print(f"ERROR: Could not find function")
        sys.exit(1)
    
    print(f"Function: lines {start}-{end} ({end-start} lines)")
    
    lines = asm_content.split('\n')[start-1:end]
    instructions, labels, used_regs = analyze_function(lines)
    
    print(f"Instructions: {len(instructions)}")
    print(f"Labels: {len(labels)}")
    print(f"Registers: {len(used_regs)}")
    
    hot_start, hot_end = find_hot_loop_range(instructions, labels)
    print(f"Hot loop: instructions {hot_start}-{hot_end} ({hot_end-hot_start} instructions)")
    
    # Generate output
    code = generate_inline_asm(instructions, labels, hot_start, hot_end)
    
    out_path = Path("target/llvm_generated_full.rs")
    out_path.write_text(code)
    print(f"\nGenerated: {out_path}")
    
    # Also save raw LLVM ASM
    raw_path = Path("target/llvm_raw_decode.s")
    raw_path.write_text('\n'.join(lines))
    print(f"Raw ASM: {raw_path}")
    
    # Preview
    print("\n" + "=" * 70)
    print("PREVIEW (first 60 lines):")
    print("=" * 70)
    for line in code.split('\n')[:60]:
        print(line)
    
    print(f"\n... ({len(code.split(chr(10))) - 60} more lines)")
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Review target/llvm_generated_full.rs")
    print("2. Copy to src/asm_decode.rs or integrate into existing decoder")
    print("3. Test with: cargo test --release test_asm")
    print("4. Benchmark with: cargo test --release bench_asm")

if __name__ == "__main__":
    main()
