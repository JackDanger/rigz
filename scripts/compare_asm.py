#!/usr/bin/env python3
"""
ASM Comparison Tool

Compares LLVM-generated assembly with our hand-written v4 ASM,
identifying specific differences and suggesting improvements.

Usage:
    python3 scripts/compare_asm.py
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from collections import defaultdict

@dataclass
class CodeBlock:
    """A logical block of assembly code"""
    name: str
    instructions: List[str]
    
def extract_v4_asm() -> List[str]:
    """Extract our v4 ASM from asm_decode.rs"""
    asm_file = Path("src/asm_decode.rs")
    if not asm_file.exists():
        return []
    
    content = asm_file.read_text()
    
    # Find the v4 asm! block
    in_asm = False
    asm_lines = []
    brace_depth = 0
    
    for line in content.split('\n'):
        if 'decode_huffman_asm_v4' in line:
            in_asm = True
            continue
        
        if in_asm:
            if 'asm!(' in line:
                brace_depth = 1
                continue
            
            if brace_depth > 0:
                stripped = line.strip()
                if stripped.startswith('"') and stripped.endswith('",'):
                    # Extract the instruction
                    inst = stripped[1:-2]  # Remove quotes and comma
                    asm_lines.append(inst)
                
                brace_depth += line.count('(') - line.count(')')
                if brace_depth <= 0:
                    break
    
    return asm_lines

def extract_llvm_asm() -> List[str]:
    """Extract LLVM's hot loop from the saved file"""
    raw_file = Path("target/llvm_raw_decode.s")
    if not raw_file.exists():
        return []
    
    content = raw_file.read_text()
    asm_lines = []
    in_hot_loop = False
    
    for line in content.split('\n'):
        stripped = line.strip()
        
        # Skip directives
        if stripped.startswith('.') or stripped.startswith('//') or not stripped:
            continue
        
        # Start at LBB*_6 or _7 (hot loop entry)
        if re.match(r'LBB\d+_[67]:$', stripped):
            in_hot_loop = True
        
        # End at error handling
        if re.match(r'LBB\d+_12[0-9]:$', stripped):
            in_hot_loop = False
            break
        
        if in_hot_loop:
            if stripped.endswith(':'):
                asm_lines.append(f'# {stripped}')  # Mark labels
            else:
                asm_lines.append(stripped)
    
    return asm_lines

def categorize_instruction(inst: str) -> str:
    """Categorize an instruction by type"""
    opcode = inst.split()[0].lower() if inst.split() else ""
    
    if opcode in ['ldr', 'ldp', 'ldrb', 'ldrh']:
        return 'LOAD'
    elif opcode in ['str', 'stp', 'strb', 'strh']:
        return 'STORE'
    elif opcode in ['b', 'b.eq', 'b.ne', 'b.hi', 'b.lo', 'b.ls', 'b.hs', 'b.lt', 'b.gt', 'b.le', 'b.ge', 
                    'cbz', 'cbnz', 'tbz', 'tbnz', 'bl', 'ret']:
        return 'BRANCH'
    elif opcode in ['and', 'orr', 'eor', 'bic', 'mvn', 'neg']:
        return 'LOGIC'
    elif opcode in ['add', 'sub', 'mul', 'madd', 'msub']:
        return 'ARITH'
    elif opcode in ['lsl', 'lsr', 'asr', 'ror', 'ubfx', 'sbfx', 'bfxil', 'bfi']:
        return 'SHIFT'
    elif opcode in ['cmp', 'cmn', 'tst', 'ccmp']:
        return 'CMP'
    elif opcode in ['csel', 'cset', 'csinc', 'csinv', 'csneg']:
        return 'CSEL'
    elif opcode in ['mov', 'movz', 'movn', 'movk']:
        return 'MOV'
    elif opcode.startswith('v') or opcode in ['ldp', 'stp'] and 'q' in inst:
        return 'SIMD'
    else:
        return 'OTHER'

def analyze_patterns(asm_lines: List[str]) -> Dict[str, any]:
    """Analyze assembly patterns"""
    stats = {
        'total': len(asm_lines),
        'categories': defaultdict(int),
        'opcodes': defaultdict(int),
        'register_usage': defaultdict(int),
        'memory_ops': [],
        'branches': [],
    }
    
    for line in asm_lines:
        if line.startswith('#'):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        opcode = parts[0].lower()
        stats['opcodes'][opcode] += 1
        stats['categories'][categorize_instruction(line)] += 1
        
        # Track register usage
        for match in re.findall(r'\b([xwq]\d+)\b', line):
            stats['register_usage'][match] += 1
        
        # Track memory operations
        if categorize_instruction(line) == 'LOAD':
            stats['memory_ops'].append(('LOAD', line))
        elif categorize_instruction(line) == 'STORE':
            stats['memory_ops'].append(('STORE', line))
        
        # Track branches
        if categorize_instruction(line) == 'BRANCH':
            stats['branches'].append(line)
    
    return stats

def find_llvm_unique_patterns(llvm_stats: Dict, v4_stats: Dict) -> List[str]:
    """Find patterns LLVM uses that we don't"""
    unique = []
    
    llvm_ops = set(llvm_stats['opcodes'].keys())
    v4_ops = set(v4_stats['opcodes'].keys())
    
    missing = llvm_ops - v4_ops
    if missing:
        unique.append(f"Instructions LLVM uses that v4 doesn't: {', '.join(sorted(missing))}")
    
    # Check for specific patterns
    if llvm_stats['opcodes'].get('csel', 0) > 0 and v4_stats['opcodes'].get('csel', 0) == 0:
        unique.append(f"LLVM uses CSEL (conditional select) {llvm_stats['opcodes']['csel']}x - consider for branchless code")
    
    if llvm_stats['opcodes'].get('ccmp', 0) > 0 and v4_stats['opcodes'].get('ccmp', 0) == 0:
        unique.append(f"LLVM uses CCMP (conditional compare) {llvm_stats['opcodes']['ccmp']}x - for chained conditions")
    
    if llvm_stats['opcodes'].get('bfxil', 0) > 0 and v4_stats['opcodes'].get('bfxil', 0) == 0:
        unique.append(f"LLVM uses BFXIL (bitfield insert low) {llvm_stats['opcodes']['bfxil']}x - for bit manipulation")
    
    return unique

def generate_improvement_suggestions(llvm_lines: List[str], v4_lines: List[str]) -> List[str]:
    """Generate specific improvement suggestions"""
    suggestions = []
    
    llvm_stats = analyze_patterns(llvm_lines)
    v4_stats = analyze_patterns(v4_lines)
    
    # Compare sizes
    suggestions.append(f"Hot loop size: LLVM={llvm_stats['total']} vs v4={v4_stats['total']} instructions")
    
    # Compare instruction mix
    for cat in ['LOAD', 'STORE', 'BRANCH', 'CSEL', 'SIMD']:
        llvm_count = llvm_stats['categories'][cat]
        v4_count = v4_stats['categories'][cat]
        if llvm_count != v4_count:
            diff = llvm_count - v4_count
            if diff > 0:
                suggestions.append(f"LLVM uses {diff} more {cat} instructions")
            else:
                suggestions.append(f"v4 uses {-diff} more {cat} instructions")
    
    # Find LLVM-unique patterns
    unique = find_llvm_unique_patterns(llvm_stats, v4_stats)
    suggestions.extend(unique)
    
    return suggestions

def extract_refill_pattern(asm_lines: List[str]) -> List[str]:
    """Extract the refill pattern from assembly"""
    refill = []
    in_refill = False
    
    for line in asm_lines:
        # Look for refill check pattern
        if 'cmp' in line.lower() and '47' in line:  # Compare bitsleft to 47
            in_refill = True
        
        if in_refill:
            refill.append(line)
            if 'orr' in line.lower() and len(refill) > 5:
                in_refill = False
                break
    
    return refill

def extract_literal_pattern(asm_lines: List[str]) -> List[str]:
    """Extract the literal decode pattern from assembly"""
    literal = []
    in_literal = False
    
    for line in asm_lines:
        # Look for literal check pattern (tbnz/tbz with bit 31)
        if ('tbnz' in line.lower() or 'tbz' in line.lower()) and '#31' in line:
            in_literal = True
        
        if in_literal:
            literal.append(line)
            if 'strb' in line.lower():  # End at store byte
                in_literal = False
                break
    
    return literal

def main():
    print("=" * 70)
    print("ASM Comparison Tool - LLVM vs v4")
    print("=" * 70)
    
    # Extract both ASM versions
    v4_asm = extract_v4_asm()
    llvm_asm = extract_llvm_asm()
    
    if not v4_asm:
        print("ERROR: Could not extract v4 ASM from src/asm_decode.rs")
        sys.exit(1)
    
    if not llvm_asm:
        print("ERROR: Could not extract LLVM ASM. Run llvm_to_inline_asm.py first.")
        sys.exit(1)
    
    print(f"\nExtracted:")
    print(f"  v4 ASM: {len(v4_asm)} instructions")
    print(f"  LLVM ASM: {len(llvm_asm)} instructions")
    
    # Analyze patterns
    v4_stats = analyze_patterns(v4_asm)
    llvm_stats = analyze_patterns(llvm_asm)
    
    print("\n" + "=" * 70)
    print("INSTRUCTION CATEGORY COMPARISON")
    print("=" * 70)
    
    categories = sorted(set(v4_stats['categories'].keys()) | set(llvm_stats['categories'].keys()))
    print(f"{'Category':<12} {'LLVM':>8} {'v4':>8} {'Diff':>8}")
    print("-" * 40)
    for cat in categories:
        llvm_c = llvm_stats['categories'][cat]
        v4_c = v4_stats['categories'][cat]
        diff = llvm_c - v4_c
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{cat:<12} {llvm_c:>8} {v4_c:>8} {diff_str:>8}")
    
    print("\n" + "=" * 70)
    print("TOP 15 OPCODES COMPARISON")
    print("=" * 70)
    
    all_ops = set(llvm_stats['opcodes'].keys()) | set(v4_stats['opcodes'].keys())
    sorted_ops = sorted(all_ops, key=lambda o: llvm_stats['opcodes'].get(o, 0) + v4_stats['opcodes'].get(o, 0), reverse=True)[:15]
    
    print(f"{'Opcode':<12} {'LLVM':>8} {'v4':>8}")
    print("-" * 30)
    for op in sorted_ops:
        print(f"{op:<12} {llvm_stats['opcodes'].get(op, 0):>8} {v4_stats['opcodes'].get(op, 0):>8}")
    
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 70)
    
    suggestions = generate_improvement_suggestions(llvm_asm, v4_asm)
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s}")
    
    print("\n" + "=" * 70)
    print("LLVM REFILL PATTERN")
    print("=" * 70)
    
    refill = extract_refill_pattern(llvm_asm)
    for line in refill[:15]:
        print(f"  {line}")
    if len(refill) > 15:
        print(f"  ... ({len(refill) - 15} more lines)")
    
    print("\n" + "=" * 70)
    print("LLVM LITERAL PATTERN")
    print("=" * 70)
    
    literal = extract_literal_pattern(llvm_asm)
    for line in literal[:15]:
        print(f"  {line}")
    if len(literal) > 15:
        print(f"  ... ({len(literal) - 15} more lines)")
    
    # Find unique LLVM opcodes
    llvm_only = set(llvm_stats['opcodes'].keys()) - set(v4_stats['opcodes'].keys())
    if llvm_only:
        print("\n" + "=" * 70)
        print("OPCODES IN LLVM BUT NOT IN v4")
        print("=" * 70)
        for op in sorted(llvm_only):
            count = llvm_stats['opcodes'][op]
            print(f"  {op}: {count}x")
    
    # Generate a summary of what to implement
    print("\n" + "=" * 70)
    print("ACTION ITEMS")
    print("=" * 70)
    print("""
Based on analysis, here are specific improvements to make:

1. INSTRUCTION SELECTION:
   - Use CCMP for chained conditions (avoids extra branches)
   - Use CSEL for branchless value selection
   - Use BFXIL for bit manipulation

2. REGISTER ALLOCATION:
   - LLVM keeps constants in registers (x16=7, x17=-1, x20=199)
   - This avoids repeated immediate loads

3. LOOP STRUCTURE:
   - LLVM's hot loop has {llvm_size} instructions vs v4's {v4_size}
   - LLVM processes multiple literals before checking bounds
   
4. MEMORY ACCESS:
   - LLVM uses more aggressive prefetching
   - Consider using PRFM for data prefetch
""".format(llvm_size=llvm_stats['total'], v4_size=v4_stats['total']))

if __name__ == "__main__":
    main()
