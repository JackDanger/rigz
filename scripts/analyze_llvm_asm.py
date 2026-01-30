#!/usr/bin/env python3
"""
LLVM Assembly Analyzer for gzippy

This script extracts and analyzes LLVM's generated assembly for the Huffman
decode hot loop, comparing it with our hand-written ASM to identify
optimization opportunities.

Usage:
    python3 scripts/analyze_llvm_asm.py

Prerequisites:
    RUSTFLAGS="--emit asm" cargo build --release
"""

import os
import re
import sys
import subprocess
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

@dataclass
class Instruction:
    """Represents a single assembly instruction"""
    address: str
    opcode: str
    operands: str
    raw: str
    line_num: int
    
    @property
    def is_load(self) -> bool:
        return self.opcode in ('ldr', 'ldp', 'ldrb', 'ldrh', 'ldrsw', 'ldur')
    
    @property
    def is_store(self) -> bool:
        return self.opcode in ('str', 'stp', 'strb', 'strh', 'stur')
    
    @property
    def is_branch(self) -> bool:
        return self.opcode.startswith('b') or self.opcode in ('ret', 'cbz', 'cbnz', 'tbz', 'tbnz')
    
    @property
    def is_simd(self) -> bool:
        return any(r in self.operands for r in ('q0', 'q1', 'q2', 'q3', 'v0', 'v1', 'v2', 'v3'))
    
    @property
    def dest_reg(self) -> Optional[str]:
        """Extract destination register"""
        if self.operands:
            parts = self.operands.split(',')
            if parts:
                reg = parts[0].strip()
                if reg.startswith(('x', 'w', 'q', 'v')):
                    return reg
        return None
    
    @property
    def src_regs(self) -> List[str]:
        """Extract source registers"""
        regs = []
        parts = self.operands.split(',')[1:] if ',' in self.operands else []
        for part in parts:
            # Extract register references
            for match in re.findall(r'\b([xwqv]\d+)\b', part):
                regs.append(match)
        return regs

@dataclass
class BasicBlock:
    """A sequence of instructions ending in a branch"""
    label: str
    instructions: List[Instruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.instructions)

@dataclass 
class AnalysisResult:
    """Results of analyzing a function"""
    function_name: str
    total_instructions: int
    blocks: List[BasicBlock]
    instruction_counts: Counter
    register_usage: Dict[str, int]
    load_count: int
    store_count: int
    branch_count: int
    simd_count: int
    dependency_chains: List[List[str]]

def find_asm_file() -> Optional[Path]:
    """Find the generated .s file"""
    deps_dir = Path("target/release/deps")
    if not deps_dir.exists():
        return None
    
    for f in deps_dir.glob("gzippy-*.s"):
        return f
    return None

def generate_asm():
    """Generate LLVM assembly"""
    print("Generating LLVM assembly...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        env={**os.environ, "RUSTFLAGS": "--emit asm"},
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        sys.exit(1)
    print("Assembly generated successfully")

def parse_instruction(line: str, line_num: int) -> Optional[Instruction]:
    """Parse a single assembly line"""
    line = line.strip()
    if not line or line.startswith(';') or line.startswith('.'):
        return None
    if line.endswith(':'):  # Label
        return None
    
    # Skip directives
    if line.startswith(('.', '//', '@')):
        return None
        
    # Parse instruction
    parts = line.split(None, 1)
    if not parts:
        return None
    
    opcode = parts[0].lower()
    operands = parts[1] if len(parts) > 1 else ""
    
    # Remove comments
    if '//' in operands:
        operands = operands.split('//')[0].strip()
    if ';' in operands:
        operands = operands.split(';')[0].strip()
    
    return Instruction(
        address="",
        opcode=opcode,
        operands=operands,
        raw=line,
        line_num=line_num
    )

def extract_function(asm_content: str, func_name: str) -> Tuple[List[str], int]:
    """Extract a function's assembly from the file"""
    lines = asm_content.split('\n')
    
    # Find function start - look for mangled Rust names
    start_idx = None
    for i, line in enumerate(lines):
        # Check for the mangled function name pattern
        if 'decode_huffman_libdeflate_style' in line and ':' in line:
            start_idx = i
            break
        if func_name in line and ':' in line:
            start_idx = i
            break
    
    if start_idx is None:
        return [], 0
    
    # Find function end - look for next function label or end marker
    end_idx = len(lines)
    for i in range(start_idx + 1, min(start_idx + 2000, len(lines))):
        line = lines[i].strip()
        
        # Skip local labels (LBB*, Ltmp*, etc)
        if line.startswith('LBB') or line.startswith('Ltmp') or line.startswith('Lfunc'):
            continue
        if line.startswith('.'):
            continue
            
        # End if we hit a new global function (starts with _ or letter, ends with :)
        if line.endswith(':') and (line.startswith('_') or line[0].isalpha()):
            # But not if it's a local label
            if not any(line.startswith(p) for p in ('LBB', 'Ltmp', 'Lfunc', '.L')):
                end_idx = i
                break
    
    return lines[start_idx:end_idx], start_idx

def analyze_function(lines: List[str], func_name: str) -> AnalysisResult:
    """Analyze a function's assembly"""
    instructions = []
    blocks = []
    current_block = BasicBlock(label="entry")
    
    instruction_counts = Counter()
    register_usage = defaultdict(int)
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Handle labels
        if line.endswith(':') and not line.startswith('.'):
            if current_block.instructions:
                blocks.append(current_block)
            current_block = BasicBlock(label=line[:-1])
            continue
        
        inst = parse_instruction(line, i)
        if inst:
            instructions.append(inst)
            current_block.instructions.append(inst)
            instruction_counts[inst.opcode] += 1
            
            # Track register usage
            if inst.dest_reg:
                register_usage[inst.dest_reg] += 1
            for reg in inst.src_regs:
                register_usage[reg] += 1
            
            # Track block successors
            if inst.is_branch:
                # Extract branch target
                for match in re.findall(r'\b(LBB\d+_\d+|\.L\w+)\b', inst.operands):
                    current_block.successors.append(match)
    
    if current_block.instructions:
        blocks.append(current_block)
    
    # Calculate statistics
    load_count = sum(1 for i in instructions if i.is_load)
    store_count = sum(1 for i in instructions if i.is_store)
    branch_count = sum(1 for i in instructions if i.is_branch)
    simd_count = sum(1 for i in instructions if i.is_simd)
    
    # Analyze dependency chains
    dependency_chains = analyze_dependencies(instructions[:100])  # First 100 instructions
    
    return AnalysisResult(
        function_name=func_name,
        total_instructions=len(instructions),
        blocks=blocks,
        instruction_counts=instruction_counts,
        register_usage=dict(register_usage),
        load_count=load_count,
        store_count=store_count,
        branch_count=branch_count,
        simd_count=simd_count,
        dependency_chains=dependency_chains
    )

def analyze_dependencies(instructions: List[Instruction]) -> List[List[str]]:
    """Find dependency chains in instructions"""
    chains = []
    current_chain = []
    last_dest = None
    
    for inst in instructions:
        if last_dest and last_dest in inst.src_regs:
            current_chain.append(f"{inst.opcode} {inst.operands}")
        else:
            if len(current_chain) > 2:
                chains.append(current_chain)
            current_chain = [f"{inst.opcode} {inst.operands}"]
        
        last_dest = inst.dest_reg
    
    if len(current_chain) > 2:
        chains.append(current_chain)
    
    return chains[:5]  # Top 5 chains

def extract_hot_loop(lines: List[str]) -> List[str]:
    """Extract the main decode loop (between loop start and end labels)"""
    in_loop = False
    loop_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Look for loop patterns
        if 'LBB' in stripped and '_7:' in stripped:  # Common LLVM loop start
            in_loop = True
        elif 'LBB' in stripped and ('_99:' in stripped or '_exit' in stripped.lower()):
            in_loop = False
        
        if in_loop:
            loop_lines.append(line)
    
    # If we didn't find specific labels, return first 200 lines of function body
    if not loop_lines:
        return [l for l in lines[:200] if not l.strip().startswith('.')]
    
    return loop_lines

def analyze_scheduling_quality(instructions: List[Instruction]) -> Dict[str, any]:
    """Analyze instruction scheduling quality"""
    metrics = {
        'load_use_distance': [],  # Distance between load and first use
        'back_to_back_deps': 0,   # Sequential dependent instructions
        'load_clustering': 0,      # Loads grouped together
    }
    
    load_dests = {}  # dest_reg -> instruction index
    
    for i, inst in enumerate(instructions):
        if inst.is_load and inst.dest_reg:
            load_dests[inst.dest_reg] = i
        
        for reg in inst.src_regs:
            if reg in load_dests:
                distance = i - load_dests[reg]
                metrics['load_use_distance'].append(distance)
                if distance == 1:
                    metrics['back_to_back_deps'] += 1
    
    # Analyze load clustering
    prev_was_load = False
    for inst in instructions:
        if inst.is_load:
            if prev_was_load:
                metrics['load_clustering'] += 1
            prev_was_load = True
        else:
            prev_was_load = False
    
    return metrics

def compare_with_v4(llvm_result: AnalysisResult) -> Dict[str, any]:
    """Compare LLVM's code with our v4 ASM"""
    # Read our v4 ASM
    v4_path = Path("src/asm_decode.rs")
    if not v4_path.exists():
        return {"error": "v4 source not found"}
    
    v4_content = v4_path.read_text()
    
    # Extract v4 ASM block
    v4_asm_match = re.search(
        r'pub fn decode_huffman_asm_v4.*?std::arch::asm!\((.*?)\s*options\(nostack\)',
        v4_content, re.DOTALL
    )
    
    if not v4_asm_match:
        return {"error": "v4 ASM block not found"}
    
    v4_asm = v4_asm_match.group(1)
    
    # Count v4 instructions
    v4_instructions = []
    for line in v4_asm.split('\n'):
        line = line.strip().strip('"').strip(',').strip()
        if line and not line.startswith('//') and not line.endswith(':'):
            # Parse as instruction
            parts = line.split(None, 1)
            if parts and parts[0].isalpha():
                v4_instructions.append(parts[0])
    
    v4_counts = Counter(v4_instructions)
    
    return {
        "v4_total_instructions": len(v4_instructions),
        "llvm_total_instructions": llvm_result.total_instructions,
        "ratio": llvm_result.total_instructions / max(len(v4_instructions), 1),
        "v4_instruction_counts": dict(v4_counts),
        "unique_to_llvm": [op for op in llvm_result.instruction_counts if op not in v4_counts],
        "unique_to_v4": [op for op in v4_counts if op not in llvm_result.instruction_counts],
    }

def generate_recommendations(llvm_result: AnalysisResult, comparison: Dict) -> List[str]:
    """Generate specific recommendations based on analysis"""
    recommendations = []
    
    # Instruction count
    if comparison.get("ratio", 1) > 1.5:
        recommendations.append(
            f"LLVM uses {comparison['ratio']:.1f}x more instructions. This suggests LLVM "
            "is doing more aggressive unrolling or has additional optimization paths."
        )
    
    # Load scheduling
    if llvm_result.load_count > 20:
        recommendations.append(
            f"LLVM has {llvm_result.load_count} load instructions. Check if loads are "
            "being issued early to hide memory latency."
        )
    
    # SIMD usage
    if llvm_result.simd_count > 0:
        recommendations.append(
            f"LLVM uses {llvm_result.simd_count} SIMD instructions. Verify our v4 has "
            "equivalent SIMD coverage for match copy."
        )
    
    # Top instructions
    top_ops = llvm_result.instruction_counts.most_common(10)
    recommendations.append(
        f"Top LLVM operations: {', '.join(f'{op}({cnt})' for op, cnt in top_ops)}"
    )
    
    # Unique instructions
    if comparison.get("unique_to_llvm"):
        recommendations.append(
            f"Instructions in LLVM but not v4: {', '.join(comparison['unique_to_llvm'][:10])}"
        )
    
    # Register usage
    most_used = sorted(llvm_result.register_usage.items(), key=lambda x: -x[1])[:5]
    recommendations.append(
        f"Most used registers: {', '.join(f'{r}({c})' for r, c in most_used)}"
    )
    
    return recommendations

def find_preload_patterns(instructions: List[Instruction]) -> List[str]:
    """Find patterns where LLVM preloads data before it's needed"""
    patterns = []
    
    for i, inst in enumerate(instructions[:-10]):
        if inst.opcode == 'ldr' and inst.dest_reg:
            # Look for use of this register
            dest = inst.dest_reg
            for j in range(i + 1, min(i + 10, len(instructions))):
                later = instructions[j]
                if dest in later.src_regs:
                    distance = j - i
                    if distance >= 3:
                        patterns.append(
                            f"Preload: {inst.opcode} {inst.operands} -> used {distance} insts later"
                        )
                    break
    
    return patterns[:10]

def output_analysis(llvm_result: AnalysisResult, comparison: Dict, 
                   recommendations: List[str], preload_patterns: List[str],
                   scheduling: Dict):
    """Output the analysis results"""
    print("\n" + "=" * 70)
    print("LLVM ASSEMBLY ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\n## Function: {llvm_result.function_name}")
    print(f"Total instructions: {llvm_result.total_instructions}")
    print(f"Basic blocks: {len(llvm_result.blocks)}")
    print(f"Load instructions: {llvm_result.load_count}")
    print(f"Store instructions: {llvm_result.store_count}")
    print(f"Branch instructions: {llvm_result.branch_count}")
    print(f"SIMD instructions: {llvm_result.simd_count}")
    
    print("\n## Comparison with v4 ASM")
    print(f"v4 instructions: {comparison.get('v4_total_instructions', 'N/A')}")
    print(f"LLVM instructions: {comparison.get('llvm_total_instructions', 'N/A')}")
    print(f"Ratio (LLVM/v4): {comparison.get('ratio', 'N/A'):.2f}x")
    
    print("\n## Instruction Breakdown (Top 15)")
    for op, count in llvm_result.instruction_counts.most_common(15):
        print(f"  {op:10s}: {count:4d}")
    
    print("\n## Scheduling Analysis")
    if scheduling.get('load_use_distance'):
        avg_dist = sum(scheduling['load_use_distance']) / len(scheduling['load_use_distance'])
        print(f"  Average load-to-use distance: {avg_dist:.1f} instructions")
    print(f"  Back-to-back dependencies: {scheduling.get('back_to_back_deps', 0)}")
    print(f"  Load clustering score: {scheduling.get('load_clustering', 0)}")
    
    print("\n## Preload Patterns (LLVM's Load Hiding)")
    if preload_patterns:
        for p in preload_patterns[:5]:
            print(f"  {p}")
    else:
        print("  No significant preload patterns found")
    
    print("\n## Recommendations")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    print("\n## Dependency Chains (potential bottlenecks)")
    for i, chain in enumerate(llvm_result.dependency_chains[:3], 1):
        print(f"\nChain {i} ({len(chain)} instructions):")
        for inst in chain[:5]:
            print(f"    {inst}")
        if len(chain) > 5:
            print(f"    ... and {len(chain) - 5} more")
    
    print("\n" + "=" * 70)

def extract_loop_structure(lines: List[str]) -> Dict[str, List[str]]:
    """Extract and categorize different parts of the decode loop"""
    structure = {
        'refill': [],
        'literal': [],
        'length': [],
        'distance': [],
        'match_copy': [],
        'subtable': [],
        'other': []
    }
    
    current_section = 'other'
    
    for line in lines:
        stripped = line.strip().lower()
        
        # Detect section based on comments or patterns
        if 'refill' in stripped or 'bitsleft' in stripped:
            current_section = 'refill'
        elif 'literal' in stripped or 'strb' in stripped:
            current_section = 'literal'
        elif 'length' in stripped or 'w26' in stripped or 'w15' in stripped:
            current_section = 'length'
        elif 'dist' in stripped:
            current_section = 'distance'
        elif 'copy' in stripped or 'ldp' in stripped or 'stp' in stripped:
            current_section = 'match_copy'
        elif 'subtable' in stripped:
            current_section = 'subtable'
        
        inst = parse_instruction(line, 0)
        if inst:
            structure[current_section].append(inst.raw)
    
    return structure

def main():
    print("LLVM Assembly Analyzer for gzippy")
    print("-" * 40)
    
    # Check if we need to generate assembly
    asm_file = find_asm_file()
    if not asm_file:
        generate_asm()
        asm_file = find_asm_file()
    
    if not asm_file:
        print("ERROR: Could not find or generate assembly file")
        sys.exit(1)
    
    print(f"Analyzing: {asm_file}")
    print(f"File size: {asm_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Read assembly content
    asm_content = asm_file.read_text()
    
    # Find and extract the decode function
    func_name = "decode_huffman_libdeflate_style"
    lines, start_line = extract_function(asm_content, func_name)
    
    if not lines:
        print(f"ERROR: Could not find function {func_name}")
        print("Searching for alternative patterns...")
        
        # Try to find any decode function
        for pattern in ['decode_huffman', 'inflate_into', 'decode_block']:
            for i, line in enumerate(asm_content.split('\n')):
                if pattern in line and ':' in line and not line.strip().startswith('.'):
                    print(f"Found potential function at line {i}: {line[:80]}...")
        sys.exit(1)
    
    print(f"Found function at line {start_line}, {len(lines)} lines")
    
    # Analyze the function
    result = analyze_function(lines, func_name)
    
    # Extract hot loop
    hot_loop_lines = extract_hot_loop(lines)
    print(f"Hot loop: ~{len(hot_loop_lines)} lines")
    
    # Analyze scheduling
    loop_instructions = [parse_instruction(l, i) for i, l in enumerate(hot_loop_lines)]
    loop_instructions = [i for i in loop_instructions if i]
    scheduling = analyze_scheduling_quality(loop_instructions)
    
    # Find preload patterns
    preload_patterns = find_preload_patterns(loop_instructions)
    
    # Compare with v4
    comparison = compare_with_v4(result)
    
    # Generate recommendations
    recommendations = generate_recommendations(result, comparison)
    
    # Output results
    output_analysis(result, comparison, recommendations, preload_patterns, scheduling)
    
    # Extract loop structure
    structure = extract_loop_structure(hot_loop_lines)
    print("\n## Loop Structure Breakdown")
    for section, insts in structure.items():
        if insts:
            print(f"  {section}: {len(insts)} instructions")
    
    # Save detailed output
    output_path = Path("target/llvm_analysis.txt")
    with open(output_path, 'w') as f:
        f.write(f"LLVM Analysis for {func_name}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("## Full Instruction List\n\n")
        for block in result.blocks[:10]:  # First 10 blocks
            f.write(f"\n{block.label}:\n")
            for inst in block.instructions[:50]:  # First 50 per block
                f.write(f"  {inst.raw}\n")
        
        f.write("\n\n## All Instruction Counts\n\n")
        for op, count in sorted(result.instruction_counts.items()):
            f.write(f"{op}: {count}\n")
        
        f.write("\n\n## Register Usage\n\n")
        for reg, count in sorted(result.register_usage.items(), key=lambda x: -x[1]):
            f.write(f"{reg}: {count}\n")
    
    print(f"\nDetailed analysis saved to: {output_path}")
    
    # Generate actionable diff
    print("\n## ACTIONABLE ITEMS TO CLOSE THE GAP")
    print("-" * 40)
    
    # Check for specific patterns
    llvm_ops = set(result.instruction_counts.keys())
    v4_ops = set(comparison.get('v4_instruction_counts', {}).keys())
    
    if 'prfm' in llvm_ops and 'prfm' not in v4_ops:
        print("1. Add PREFETCH instructions (prfm) - LLVM prefetches data")
    
    if scheduling.get('load_use_distance') and sum(scheduling['load_use_distance']) / len(scheduling['load_use_distance']) > 3:
        print("2. Improve load scheduling - LLVM has better load-use distance")
    
    if result.simd_count > 10:
        print("3. Verify SIMD coverage - LLVM uses significant SIMD")
    
    if len(result.blocks) > 20:
        print("4. Consider loop structure - LLVM has many basic blocks (unrolling)")
    
    print("\nRun with DEBUG=1 for even more details")

if __name__ == "__main__":
    main()
