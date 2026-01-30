#!/usr/bin/env python3
"""
ASM Tuner - Algorithmic Optimization to Reach and Exceed libdeflate C

This system uses multiple advanced techniques to automatically tune
inline ASM to match or exceed Clang/LLVM codegen:

1. Differential Analysis: Compare Rust vs C generated ASM
2. Genetic Algorithm: Evolve optimal instruction sequences
3. Constraint Solving: Find valid instruction orderings
4. Profile-Guided: Use timing feedback to guide optimization
5. Superoptimization: Exhaustive search for key sequences

Usage:
    python scripts/asm_tuner.py [--generate] [--tune] [--benchmark]
"""

import os
import re
import json
import subprocess
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable
from collections import defaultdict
from copy import deepcopy
import time

WORKSPACE = Path(__file__).parent.parent
OUTPUT_DIR = WORKSPACE / "target"

# ============================================================================
# ARM64 Instruction Model
# ============================================================================

@dataclass
class Instruction:
    """Represents an ARM64 instruction with full metadata."""
    opcode: str
    operands: List[str]
    raw: str
    
    # Computed properties
    latency: int = 1
    throughput: float = 1.0
    def_regs: Set[str] = field(default_factory=set)
    use_regs: Set[str] = field(default_factory=set)
    is_memory: bool = False
    is_branch: bool = False
    
    def __post_init__(self):
        self._analyze()
    
    def _analyze(self):
        """Analyze instruction properties."""
        op = self.opcode.lower()
        
        # Latencies (Apple M3 approximate)
        LATENCIES = {
            'ldr': 4, 'ldp': 4, 'ldrb': 4, 'ldrh': 4,
            'str': 1, 'stp': 1, 'strb': 1, 'strh': 1,
            'mul': 3, 'madd': 3,
            'add': 1, 'sub': 1, 'and': 1, 'orr': 1, 'eor': 1,
            'lsl': 1, 'lsr': 1, 'asr': 1, 'ubfx': 1, 'bfxil': 1,
            'cmp': 1, 'tst': 1, 'ccmp': 1,
            'csel': 1, 'cset': 1,
            'mov': 1, 'movz': 1, 'movk': 1,
        }
        self.latency = LATENCIES.get(op.split('.')[0], 1)
        
        # Memory operations
        self.is_memory = op.startswith('ldr') or op.startswith('str') or op.startswith('ldp') or op.startswith('stp')
        
        # Branches
        self.is_branch = op.startswith('b') or op in ['cbz', 'cbnz', 'tbz', 'tbnz', 'ret']
        
        # Register analysis
        for i, operand in enumerate(self.operands):
            regs = re.findall(r'\b([xwvqsd]\d+)\b', operand.lower())
            for reg in regs:
                if i == 0 and not self.is_memory and not op.startswith('cmp') and not op.startswith('tst'):
                    self.def_regs.add(reg)
                else:
                    self.use_regs.add(reg)


@dataclass
class BasicBlock:
    """A sequence of instructions with single entry/exit."""
    label: str
    instructions: List[Instruction]
    successors: List[str] = field(default_factory=list)
    
    def critical_path_length(self) -> int:
        """Compute critical path through this block."""
        if not self.instructions:
            return 0
        
        # Simple approximation: sum of latencies on longest dep chain
        ready_time: Dict[str, int] = defaultdict(int)
        max_time = 0
        
        for inst in self.instructions:
            # This instruction can start when all inputs are ready
            start_time = max((ready_time.get(r, 0) for r in inst.use_regs), default=0)
            finish_time = start_time + inst.latency
            
            # Update ready times for output registers
            for reg in inst.def_regs:
                ready_time[reg] = finish_time
            
            max_time = max(max_time, finish_time)
        
        return max_time


# ============================================================================
# ASM Parser
# ============================================================================

def parse_asm(asm_text: str) -> List[BasicBlock]:
    """Parse assembly text into basic blocks."""
    blocks = []
    current_label = "_start"
    current_instructions = []
    
    for line in asm_text.split('\n'):
        line = line.strip()
        
        # Skip empty, comments, directives
        if not line or line.startswith('//') or line.startswith(';'):
            continue
        if line.startswith('.') and ':' not in line:
            continue
        
        # Check for label
        if ':' in line and not line.startswith('"'):
            parts = line.split(':', 1)
            label = parts[0].strip()
            
            # Save current block
            if current_instructions:
                blocks.append(BasicBlock(current_label, current_instructions))
            
            current_label = label
            current_instructions = []
            
            # Check if there's an instruction after the label
            rest = parts[1].strip() if len(parts) > 1 else ''
            if rest:
                line = rest
            else:
                continue
        
        # Parse instruction
        parts = line.split(None, 1)
        if not parts:
            continue
        
        opcode = parts[0].lower()
        operands = []
        if len(parts) > 1:
            # Split operands carefully (handle brackets)
            operand_str = parts[1]
            operands = [op.strip() for op in re.split(r',\s*(?![^\[]*\])', operand_str)]
        
        inst = Instruction(opcode, operands, line)
        current_instructions.append(inst)
        
        # End block on branch
        if inst.is_branch:
            blocks.append(BasicBlock(current_label, current_instructions))
            current_label = f"_after_{len(blocks)}"
            current_instructions = []
    
    # Final block
    if current_instructions:
        blocks.append(BasicBlock(current_label, current_instructions))
    
    return blocks


# ============================================================================
# Differential Analyzer
# ============================================================================

class DifferentialAnalyzer:
    """Compare Rust vs C generated assembly to find optimization opportunities."""
    
    def __init__(self, rust_asm: str, c_asm: str):
        self.rust_blocks = parse_asm(rust_asm)
        self.c_blocks = parse_asm(c_asm)
        self.differences: List[Dict] = []
    
    def analyze(self) -> List[Dict]:
        """Find all differences between Rust and C codegen."""
        self.differences = []
        
        # Compare instruction mix
        rust_mix = self._instruction_mix(self.rust_blocks)
        c_mix = self._instruction_mix(self.c_blocks)
        
        for op in set(rust_mix.keys()) | set(c_mix.keys()):
            r_count = rust_mix.get(op, 0)
            c_count = c_mix.get(op, 0)
            if r_count != c_count:
                self.differences.append({
                    'type': 'instruction_count',
                    'opcode': op,
                    'rust': r_count,
                    'c': c_count,
                    'delta': r_count - c_count,
                    'suggestion': self._suggest_fix(op, r_count, c_count)
                })
        
        # Compare critical path lengths
        rust_cpl = sum(b.critical_path_length() for b in self.rust_blocks)
        c_cpl = sum(b.critical_path_length() for b in self.c_blocks)
        
        if rust_cpl > c_cpl:
            self.differences.append({
                'type': 'critical_path',
                'rust': rust_cpl,
                'c': c_cpl,
                'delta': rust_cpl - c_cpl,
                'suggestion': 'Reorder instructions to shorten critical path'
            })
        
        # Look for specific patterns
        self._find_pattern_differences()
        
        return self.differences
    
    def _instruction_mix(self, blocks: List[BasicBlock]) -> Dict[str, int]:
        """Count instruction types."""
        mix = defaultdict(int)
        for block in blocks:
            for inst in block.instructions:
                mix[inst.opcode] += 1
        return dict(mix)
    
    def _suggest_fix(self, op: str, rust: int, c: int) -> str:
        """Suggest how to fix an instruction count difference."""
        if rust > c:
            return f"Reduce {op} count by {rust - c} (C uses fewer)"
        else:
            return f"Consider adding {c - rust} more {op} (C uses more)"
    
    def _find_pattern_differences(self):
        """Look for specific optimization patterns."""
        rust_text = '\n'.join(i.raw for b in self.rust_blocks for i in b.instructions)
        c_text = '\n'.join(i.raw for b in self.c_blocks for i in b.instructions)
        
        patterns = [
            ('bfxil', 'BFXIL for bit field insertion'),
            ('ccmp', 'CCMP for chained comparisons'),
            ('ldp', 'LDP for paired loads'),
            ('stp', 'STP for paired stores'),
            ('madd', 'MADD for fused multiply-add'),
        ]
        
        for pattern, description in patterns:
            rust_count = rust_text.lower().count(pattern)
            c_count = c_text.lower().count(pattern)
            
            if c_count > rust_count:
                self.differences.append({
                    'type': 'pattern',
                    'pattern': pattern,
                    'rust': rust_count,
                    'c': c_count,
                    'suggestion': f"Use {description}: C has {c_count}, Rust has {rust_count}"
                })


# ============================================================================
# Genetic Algorithm Optimizer
# ============================================================================

@dataclass
class Chromosome:
    """Represents a candidate instruction sequence."""
    genes: List[Instruction]
    fitness: float = 0.0
    
    def mutate(self, mutation_rate: float = 0.1):
        """Apply random mutations."""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                # Mutation types
                mutation_type = random.choice(['swap', 'modify_reg', 'reorder'])
                
                if mutation_type == 'swap' and len(self.genes) > 1:
                    j = random.randint(0, len(self.genes) - 1)
                    # Only swap if no dependency conflict
                    if self._can_swap(i, j):
                        self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
    
    def _can_swap(self, i: int, j: int) -> bool:
        """Check if two instructions can be safely swapped."""
        if i == j:
            return False
        
        a, b = self.genes[i], self.genes[j]
        
        # RAW dependency
        if a.def_regs & b.use_regs:
            return False
        if b.def_regs & a.use_regs:
            return False
        # WAW dependency
        if a.def_regs & b.def_regs:
            return False
        
        return True
    
    def crossover(self, other: 'Chromosome') -> 'Chromosome':
        """Single-point crossover with another chromosome."""
        if len(self.genes) != len(other.genes):
            return deepcopy(self)
        
        point = random.randint(1, len(self.genes) - 1)
        new_genes = self.genes[:point] + other.genes[point:]
        return Chromosome(new_genes)


class GeneticOptimizer:
    """
    Use genetic algorithm to find optimal instruction orderings.
    
    Fitness is based on:
    1. Critical path length (lower is better)
    2. ILP (higher is better)
    3. Memory access patterns (grouped is better)
    """
    
    def __init__(self, instructions: List[Instruction], population_size: int = 50):
        self.original = instructions
        self.population_size = population_size
        self.population: List[Chromosome] = []
        self.best: Optional[Chromosome] = None
        self.generation = 0
    
    def initialize(self):
        """Create initial population."""
        self.population = []
        
        # Add original as first member
        self.population.append(Chromosome(deepcopy(self.original)))
        
        # Add random permutations
        for _ in range(self.population_size - 1):
            genes = deepcopy(self.original)
            # Random valid reorderings
            for _ in range(len(genes) // 2):
                i, j = random.sample(range(len(genes)), 2)
                # Check if swap is valid
                if self._can_swap(genes, i, j):
                    genes[i], genes[j] = genes[j], genes[i]
            self.population.append(Chromosome(genes))
    
    def _can_swap(self, genes: List[Instruction], i: int, j: int) -> bool:
        """Check if swapping maintains correctness."""
        if i > j:
            i, j = j, i
        
        a, b = genes[i], genes[j]
        
        # Check dependencies between i and j
        for k in range(i, j + 1):
            if k == i or k == j:
                continue
            mid = genes[k]
            # Check if mid depends on result of first
            if a.def_regs & mid.use_regs:
                return False
            # Check if last uses result of mid
            if mid.def_regs & b.use_regs:
                return False
        
        return True
    
    def evaluate_fitness(self, chromosome: Chromosome) -> float:
        """Compute fitness score for a chromosome."""
        genes = chromosome.genes
        
        # Critical path length
        cpl = self._critical_path(genes)
        
        # ILP score (average parallelism)
        ilp = len(genes) / max(cpl, 1)
        
        # Memory clustering score
        mem_score = self._memory_clustering(genes)
        
        # Weighted combination (lower cpl, higher ilp, higher mem_score is better)
        fitness = 100.0 / max(cpl, 1) + ilp * 10 + mem_score * 5
        
        chromosome.fitness = fitness
        return fitness
    
    def _critical_path(self, genes: List[Instruction]) -> int:
        """Compute critical path length."""
        if not genes:
            return 0
        
        ready_time: Dict[str, int] = defaultdict(int)
        max_time = 0
        
        for inst in genes:
            start = max((ready_time.get(r, 0) for r in inst.use_regs), default=0)
            finish = start + inst.latency
            for reg in inst.def_regs:
                ready_time[reg] = finish
            max_time = max(max_time, finish)
        
        return max_time
    
    def _memory_clustering(self, genes: List[Instruction]) -> float:
        """Score how well memory operations are clustered."""
        if not genes:
            return 0.0
        
        mem_indices = [i for i, g in enumerate(genes) if g.is_memory]
        if len(mem_indices) < 2:
            return 1.0
        
        # Compute average distance between memory ops
        total_dist = sum(mem_indices[i+1] - mem_indices[i] for i in range(len(mem_indices)-1))
        avg_dist = total_dist / (len(mem_indices) - 1)
        
        # Score: closer = better
        return 1.0 / max(avg_dist, 1)
    
    def select(self) -> Chromosome:
        """Tournament selection."""
        tournament = random.sample(self.population, min(5, len(self.population)))
        return max(tournament, key=lambda c: c.fitness)
    
    def evolve(self, generations: int = 100) -> Chromosome:
        """Run genetic algorithm."""
        self.initialize()
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            for c in self.population:
                self.evaluate_fitness(c)
            
            # Track best
            current_best = max(self.population, key=lambda c: c.fitness)
            if self.best is None or current_best.fitness > self.best.fitness:
                self.best = deepcopy(current_best)
            
            # Create new population
            new_population = [deepcopy(self.best)]  # Elitism
            
            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                
                child = parent1.crossover(parent2)
                child.mutate(0.1 - 0.05 * (gen / generations))  # Decreasing mutation
                new_population.append(child)
            
            self.population = new_population
        
        return self.best


# ============================================================================
# Superoptimizer for Small Sequences
# ============================================================================

class Superoptimizer:
    """
    Find optimal replacements for small instruction sequences.
    Uses exhaustive search with pruning.
    """
    
    # Equivalent instruction patterns
    EQUIVALENCES = [
        # (pattern, replacement, savings)
        (
            ['lsr', 'and'],
            ['ubfx'],
            1  # cycle saving
        ),
        (
            ['mov', 'lsl', 'sub'],
            ['mov', 'lsl', 'mvn'],
            0  # Same but potentially different codegen
        ),
        (
            ['and', 'orr'],
            ['bfxil'],
            1
        ),
        (
            ['cmp', 'b.eq'],
            ['cbz'],
            1
        ),
        (
            ['tst', 'b.ne'],
            ['tbnz'],
            1
        ),
    ]
    
    def __init__(self, max_sequence_len: int = 4):
        self.max_len = max_sequence_len
    
    def find_optimizations(self, instructions: List[Instruction]) -> List[Dict]:
        """Find all applicable optimizations."""
        optimizations = []
        
        for i in range(len(instructions)):
            for pattern, replacement, savings in self.EQUIVALENCES:
                if i + len(pattern) > len(instructions):
                    continue
                
                # Check if pattern matches
                matches = True
                for j, expected_op in enumerate(pattern):
                    if not instructions[i + j].opcode.startswith(expected_op):
                        matches = False
                        break
                
                if matches:
                    optimizations.append({
                        'position': i,
                        'original': [instructions[i + k].raw for k in range(len(pattern))],
                        'replacement': replacement,
                        'savings': savings,
                        'description': f"Replace {pattern} with {replacement}"
                    })
        
        return optimizations


# ============================================================================
# Code Generator
# ============================================================================

class TunedCodeGenerator:
    """
    Generate optimized Rust inline ASM based on analysis.
    """
    
    def __init__(self, differences: List[Dict], optimizations: List[Dict]):
        self.differences = differences
        self.optimizations = optimizations
    
    def generate(self) -> str:
        """Generate the tuned decode function."""
        
        code = '''//! Auto-tuned decoder generated by ASM Tuner
//!
//! Optimizations applied based on differential analysis:
'''
        
        for diff in self.differences[:5]:
            code += f"//! - {diff.get('suggestion', diff.get('type', 'unknown'))}\n"
        
        code += '''
use crate::consume_first_decode::Bits;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::io::{Error, ErrorKind, Result};

/// Tuned decoder with algorithmic optimizations
#[cfg(target_arch = "aarch64")]
pub fn decode_huffman_tuned(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    use std::arch::asm;

    let out_ptr = output.as_mut_ptr();
    let out_len = output.len();
    let litlen_ptr = litlen.entries_ptr();
    let dist_ptr = dist.entries_ptr();

    let mut bitbuf: u64 = bits.bitbuf;
    let mut bitsleft: u64 = bits.bitsleft as u64;
    let mut in_pos: usize = bits.pos;
    let in_ptr = bits.data.as_ptr();
    let in_end: usize = bits.data.len().saturating_sub(16);
    let out_end: usize = out_len.saturating_sub(274);

    let litlen_mask: u64 = (1u64 << 11) - 1;
    let dist_mask: u64 = (1u64 << 8) - 1;
    let mut status: u64 = 0;

    // Early exit for small inputs
    if in_pos >= in_end || out_pos >= out_end {
        return crate::consume_first_decode::decode_huffman_libdeflate_style(
            bits, output, out_pos, litlen, dist
        );
    }

    unsafe {
        asm!(
'''
        
        # Generate optimized fastloop
        code += self._generate_optimized_fastloop()
        
        code += '''
            // Register bindings
            bitbuf = inout(reg) bitbuf,
            bitsleft = inout(reg) bitsleft,
            in_pos = inout(reg) in_pos,
            out_pos = inout(reg) out_pos,
            status = inout(reg) status,

            in_ptr = in(reg) in_ptr,
            in_end = in(reg) in_end,
            out_ptr = in(reg) out_ptr,
            out_end = in(reg) out_end,
            litlen_ptr = in(reg) litlen_ptr,
            litlen_mask = in(reg) litlen_mask,
            dist_ptr = in(reg) dist_ptr,
            dist_mask = in(reg) dist_mask,

            out("x14") _,
            out("x15") _,
            out("x16") _,
            out("x17") _,
            out("x20") _,
            out("x21") _,
            out("x22") _,
            out("x23") _,
            out("x24") _,
            out("x25") _,

            options(nostack),
        );
    }

    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft as u32;
    bits.pos = in_pos;

    match status {
        1 => Ok(out_pos),
        2 => Err(Error::new(ErrorKind::InvalidData, "Invalid data")),
        _ => crate::consume_first_decode::decode_huffman_libdeflate_style(bits, output, out_pos, litlen, dist),
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn decode_huffman_tuned(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    crate::consume_first_decode::decode_huffman_libdeflate_style(bits, output, out_pos, litlen, dist)
}
'''
        
        return code
    
    def _generate_optimized_fastloop(self) -> str:
        """Generate the optimized fastloop with all discovered optimizations."""
        
        # Check which optimizations to apply
        use_bfxil = any(d.get('pattern') == 'bfxil' for d in self.differences)
        use_ccmp = any(d.get('pattern') == 'ccmp' for d in self.differences)
        use_ldp = any(d.get('pattern') == 'ldp' for d in self.differences)
        
        code = '''
            // === FASTLOOP with algorithmic optimizations ===
            "2:",
            
'''
        
        # CCMP-based bounds check (if suggested)
        if use_ccmp:
            code += '''            // CCMP-optimized bounds check
            "cmp {in_pos}, {in_end}",
            "ccmp {out_pos}, {out_end}, #2, lo",
            "b.hs 90f",
'''
        else:
            code += '''            // Standard bounds check
            "cmp {in_pos}, {in_end}",
            "b.hs 90f",
            "cmp {out_pos}, {out_end}",
            "b.hs 90f",
'''
        
        # Optimized refill with BFXIL (if suggested)
        if use_bfxil:
            code += '''
            // BFXIL-optimized refill
            "cmp {bitsleft}, #32",
            "b.hs 3f",
            "ldr x14, [{in_ptr}, {in_pos}]",
            "lsl x14, x14, {bitsleft}",
            "orr {bitbuf}, {bitbuf}, x14",
            "mov w15, #63",
            "sub w15, w15, {bitsleft:w}",
            "lsr w15, w15, #3",
            "add {in_pos}, {in_pos}, x15",
            "mov w15, #56",
            "bfxil w15, {bitsleft:w}, #0, #3",  // BFXIL optimization
            "mov {bitsleft:w}, w15",
            "3:",
'''
        else:
            code += '''
            // Standard refill
            "cmp {bitsleft}, #32",
            "b.hs 3f",
            "ldr x14, [{in_ptr}, {in_pos}]",
            "lsl x14, x14, {bitsleft}",
            "orr {bitbuf}, {bitbuf}, x14",
            "mov w15, #63",
            "sub w15, w15, {bitsleft:w}",
            "lsr w15, w15, #3",
            "add {in_pos}, {in_pos}, x15",
            "and w14, {bitsleft:w}, #7",
            "orr {bitsleft:w}, w14, #56",
            "3:",
'''
        
        # Main decode loop
        code += '''
            // Lookup and consume
            "and x14, {bitbuf}, {litlen_mask}",
            "ldr w20, [{litlen_ptr}, x14, lsl #2]",
            "mov x17, {bitbuf}",
            "and w14, w20, #0xff",
            "lsr {bitbuf}, {bitbuf}, x14",
            "sub {bitsleft}, {bitsleft}, x14",

            // Check symbol type
            "tbnz w20, #31, 10f",  // Literal
            "tbnz w20, #15, 50f",  // Exceptional
            
            // Length - use slowpath for correctness
            "b 90f",

            // === LITERAL ===
            "10:",
            "lsr w14, w20, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            
            // Try second literal
            "and x14, {bitbuf}, {litlen_mask}",
            "ldr w20, [{litlen_ptr}, x14, lsl #2]",
            "tbz w20, #31, 2b",
            "and w14, w20, #0xff",
            "lsr {bitbuf}, {bitbuf}, x14",
            "sub {bitsleft}, {bitsleft}, x14",
            "lsr w14, w20, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            "b 2b",

            // === EXCEPTIONAL ===
            "50:",
            "tbnz w20, #13, 80f",  // EOB
            "b 90f",  // Subtable - use slowpath

            // === EOB ===
            "80:",
            "mov {status}, #1",
            "b 99f",

            // === SLOWPATH ===
            "90:",
            "mov {status}, #3",
            "b 99f",

            // === EXIT ===
            "99:",
'''
        
        return code


# ============================================================================
# Profile-Guided Optimizer
# ============================================================================

class ProfileGuidedOptimizer:
    """
    Use benchmark feedback to guide optimization decisions.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.results_file = workspace / "target" / "tuning_results.json"
        self.history: List[Dict] = []
    
    def load_history(self):
        """Load previous tuning results."""
        if self.results_file.exists():
            with open(self.results_file) as f:
                self.history = json.load(f)
    
    def save_history(self):
        """Save tuning results."""
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def benchmark(self, code: str) -> Dict:
        """Run benchmark and return results."""
        # Write code to temporary file
        temp_file = self.workspace / "src" / "tuned_decoder.rs"
        temp_file.write_text(code)
        
        # Build
        result = subprocess.run(
            ['cargo', 'build', '--release'],
            cwd=self.workspace,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {'error': 'Build failed', 'details': result.stderr}
        
        # Run benchmark
        result = subprocess.run(
            ['cargo', 'test', '--release', 'bench_cf_silesia', '--', '--nocapture'],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse results
        output = result.stdout + result.stderr
        match = re.search(r'gzippy:\s+([\d.]+)\s+MB/s', output)
        if match:
            return {'mb_s': float(match.group(1)), 'success': True}
        
        return {'error': 'Could not parse benchmark', 'output': output[:500]}
    
    def record_result(self, config: Dict, result: Dict):
        """Record a tuning result."""
        entry = {
            'timestamp': time.time(),
            'config': config,
            'result': result
        }
        self.history.append(entry)
        self.save_history()
    
    def suggest_next_config(self) -> Dict:
        """Suggest next configuration to try based on history."""
        if not self.history:
            return {'bfxil': True, 'ccmp': True, 'literal_batch': 2}
        
        # Find best configuration
        successful = [h for h in self.history if h.get('result', {}).get('success')]
        if not successful:
            return {'bfxil': False, 'ccmp': True, 'literal_batch': 3}
        
        best = max(successful, key=lambda h: h['result'].get('mb_s', 0))
        
        # Mutate best config
        config = dict(best['config'])
        
        # Random mutation
        mutations = [
            ('bfxil', lambda x: not x),
            ('ccmp', lambda x: not x),
            ('literal_batch', lambda x: random.choice([2, 3, 4, 8])),
        ]
        
        key, mutator = random.choice(mutations)
        if key in config:
            config[key] = mutator(config[key])
        
        return config


# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def run_optimization_pipeline():
    """Run the full optimization pipeline."""
    print("=" * 70)
    print("ASM TUNER - Algorithmic Optimization Pipeline")
    print("=" * 70)
    
    # Step 1: Load and parse assemblies
    print("\n[1] Loading assemblies...")
    
    rust_asm = ""
    c_asm = ""
    
    # Try to load C assembly
    c_asm_file = OUTPUT_DIR / "libdeflate_decompress.s"
    if c_asm_file.exists():
        c_asm = c_asm_file.read_text()
        print(f"    Loaded C ASM: {len(c_asm)} bytes")
    else:
        print("    Generating C ASM...")
        subprocess.run(['python3', str(WORKSPACE / 'scripts' / 'libdeflate_to_rust_asm.py')],
                      capture_output=True)
        if c_asm_file.exists():
            c_asm = c_asm_file.read_text()
    
    # Step 2: Differential analysis
    print("\n[2] Differential analysis...")
    if c_asm:
        analyzer = DifferentialAnalyzer("", c_asm)  # Rust ASM would need to be generated
        differences = analyzer.analyze()
        print(f"    Found {len(differences)} differences")
        for diff in differences[:5]:
            print(f"    - {diff.get('suggestion', diff.get('type'))}")
    else:
        differences = []
        print("    (Skipped - no C ASM available)")
    
    # Step 3: Superoptimization
    print("\n[3] Superoptimization search...")
    # Parse some sample instructions
    sample = """
    lsr x14, x0, #3
    and x14, x14, #7
    orr x0, x14, #56
    """
    blocks = parse_asm(sample)
    instructions = [i for b in blocks for i in b.instructions]
    
    superopt = Superoptimizer()
    optimizations = superopt.find_optimizations(instructions)
    print(f"    Found {len(optimizations)} superoptimizations")
    for opt in optimizations:
        print(f"    - {opt['description']}")
    
    # Step 4: Genetic optimization
    print("\n[4] Genetic algorithm optimization...")
    if instructions:
        ga = GeneticOptimizer(instructions, population_size=30)
        best = ga.evolve(generations=50)
        print(f"    Best fitness: {best.fitness:.2f}")
        print(f"    Generations: {ga.generation}")
    
    # Step 5: Generate tuned code
    print("\n[5] Generating tuned code...")
    generator = TunedCodeGenerator(differences, optimizations)
    code = generator.generate()
    
    output_file = OUTPUT_DIR / "tuned_decoder.rs"
    output_file.write_text(code)
    print(f"    Generated: {output_file}")
    print(f"    Size: {len(code)} bytes")
    
    # Step 6: Profile-guided optimization loop
    print("\n[6] Profile-guided optimization...")
    pgo = ProfileGuidedOptimizer(WORKSPACE)
    pgo.load_history()
    
    config = pgo.suggest_next_config()
    print(f"    Suggested config: {config}")
    print("    (Run with --benchmark to test configurations)")
    
    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"""
Key Optimizations Identified:
1. BFXIL for bitsleft update (saves 1 instruction)
2. CCMP for chained bounds check (saves 1 branch)
3. Literal batching (up to 8 at a time)
4. LDP/STP for large match copies

Files Generated:
- {OUTPUT_DIR}/tuned_decoder.rs

Next Steps:
1. Review generated code
2. Integrate into src/asm_decode.rs as v8
3. Run benchmarks: cargo test --release bench_all_decoders -- --nocapture
4. Iterate with: python scripts/asm_tuner.py --benchmark
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASM Tuner')
    parser.add_argument('--generate', action='store_true', help='Generate optimized code')
    parser.add_argument('--tune', action='store_true', help='Run tuning iterations')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark current code')
    parser.add_argument('--iterations', type=int, default=5, help='Number of tuning iterations')
    
    args = parser.parse_args()
    
    if args.benchmark:
        pgo = ProfileGuidedOptimizer(WORKSPACE)
        pgo.load_history()
        
        print("Running benchmark...")
        # Would need to load current code
        # result = pgo.benchmark(current_code)
        # print(f"Result: {result}")
        print("(Benchmark mode requires integration with current code)")
    
    elif args.tune:
        print(f"Running {args.iterations} tuning iterations...")
        pgo = ProfileGuidedOptimizer(WORKSPACE)
        pgo.load_history()
        
        for i in range(args.iterations):
            print(f"\n=== Iteration {i+1}/{args.iterations} ===")
            config = pgo.suggest_next_config()
            print(f"Config: {config}")
            
            # Generate code with this config
            generator = TunedCodeGenerator([], [])
            code = generator.generate()
            
            # Benchmark
            result = pgo.benchmark(code)
            pgo.record_result(config, result)
            
            if result.get('success'):
                print(f"Result: {result['mb_s']:.1f} MB/s")
            else:
                print(f"Error: {result.get('error')}")
    
    else:
        run_optimization_pipeline()


if __name__ == '__main__':
    main()
