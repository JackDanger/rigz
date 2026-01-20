#!/usr/bin/env python3
"""
Collect system information for benchmark reports.

Usage:
    python3 scripts/collect_system_info.py --output results/system.json
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def run_cmd(cmd: list) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_cpu_model() -> str:
    """Get CPU model name."""
    system = platform.system()
    
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        return run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    
    return platform.processor() or "Unknown"


def get_core_count() -> int:
    """Get number of CPU cores."""
    return os.cpu_count() or 1


def get_simd_features() -> str:
    """Detect available SIMD features."""
    system = platform.system()
    features = []
    
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read().lower()
                if "avx512" in content:
                    features.append("avx512")
                if "avx2" in content:
                    features.append("avx2")
                if "avx " in content:
                    features.append("avx")
                if "sse4_2" in content:
                    features.append("sse4.2")
                if "bmi2" in content:
                    features.append("bmi2")
                if "neon" in content:
                    features.append("neon")
        except Exception:
            pass
    elif system == "Darwin":
        # Check for ARM (Apple Silicon has NEON)
        machine = platform.machine()
        if machine == "arm64":
            features.append("neon")
        else:
            # Intel Mac - check features
            leaf7 = run_cmd(["sysctl", "-n", "machdep.cpu.leaf7_features"])
            features_str = run_cmd(["sysctl", "-n", "machdep.cpu.features"])
            combined = (leaf7 + " " + features_str).upper()
            
            if "AVX512" in combined:
                features.append("avx512")
            if "AVX2" in combined:
                features.append("avx2")
            if "AVX" in combined and "AVX2" not in combined:
                features.append("avx")
            if "SSE4.2" in combined or "SSE4_2" in combined:
                features.append("sse4.2")
            if "BMI2" in combined:
                features.append("bmi2")
    
    return " ".join(features) if features else "unknown"


def get_memory_gb() -> float:
    """Get total memory in GB."""
    system = platform.system()
    
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 1)
        except Exception:
            pass
    elif system == "Darwin":
        try:
            output = run_cmd(["sysctl", "-n", "hw.memsize"])
            if output:
                return round(int(output) / (1024 ** 3), 1)
        except Exception:
            pass
    
    return 0


def collect_system_info() -> dict:
    """Collect all system information."""
    return {
        "cpu": get_cpu_model(),
        "cores": get_core_count(),
        "simd": get_simd_features(),
        "memory_gb": get_memory_gb(),
        "os": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect system information")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: stdout)")
    args = parser.parse_args()
    
    info = collect_system_info()
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"System info written to {args.output}")
    else:
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
