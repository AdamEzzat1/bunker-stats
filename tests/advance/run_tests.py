#!/usr/bin/env python3
"""
Master Test Runner for bunker-stats

Provides convenient commands for running different test suites:
- Fast tests (< 5 min)
- Full benchmarks (< 30 min)
- Stress tests (1-2 hours)
- Specific modules

Usage:
    python run_tests.py fast          # Fast tests only
    python run_tests.py benchmarks    # All benchmarks
    python run_tests.py safety        # Safety tests
    python run_tests.py stress        # Stress tests
    python run_tests.py all           # Everything
    python run_tests.py module robust # Specific module
"""

import subprocess
import sys
import time
from pathlib import Path


RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_section(text):
    """Print section header"""
    print(f"\n{BOLD}{GREEN}{text}{RESET}")
    print(f"{BOLD}{GREEN}{'-'*len(text)}{RESET}")


def run_command(cmd, description):
    """Run pytest command and track time"""
    print_section(description)
    print(f"Command: {' '.join(cmd)}\n")
    
    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
    
    if result.returncode == 0:
        print(f"\n{GREEN}✓ {description} completed successfully in {elapsed:.1f}s{RESET}")
        return True
    else:
        print(f"\n{RED}✗ {description} failed after {elapsed:.1f}s{RESET}")
        return False


def run_fast_tests():
    """Run fast tests only (< 5 minutes)"""
    print_header("FAST TEST SUITE")
    
    commands = [
        (["pytest", "test_concurrency_safety.py", "-v", "--tb=short"], 
         "Concurrency Safety Tests"),
        (["pytest", "test_memory_safety.py", "-v", "--tb=short", "-m", "not slow"], 
         "Memory Safety Tests (fast)"),
    ]
    
    results = []
    for cmd, desc in commands:
        results.append(run_command(cmd, desc))
    
    return all(results)


def run_benchmarks():
    """Run all benchmarks (< 30 minutes)"""
    print_header("PERFORMANCE BENCHMARKS")
    
    commands = [
        (["pytest", "benchmarks_robust_stats.py", "-v", "--tb=short"], 
         "Robust Statistics Benchmarks"),
        (["pytest", "benchmarks_inference.py", "-v", "--tb=short"], 
         "Inference Benchmarks"),
        (["pytest", "benchmarks_resampling.py", "-v", "--tb=short"], 
         "Resampling Benchmarks"),
        (["pytest", "benchmarks_matrix.py", "-v", "--tb=short"], 
         "Matrix Operations Benchmarks"),
    ]
    
    results = []
    for cmd, desc in commands:
        results.append(run_command(cmd, desc))
    
    return all(results)


def run_safety_tests():
    """Run safety tests"""
    print_header("SAFETY TEST SUITE")
    
    commands = [
        (["pytest", "test_concurrency_safety.py", "-v", "--tb=short"], 
         "Concurrency Safety Tests"),
        (["pytest", "test_memory_safety.py", "-v", "--tb=short"], 
         "Memory Safety Tests"),
    ]
    
    results = []
    for cmd, desc in commands:
        results.append(run_command(cmd, desc))
    
    return all(results)


def run_stress_tests():
    """Run stress tests (1-2 hours)"""
    print_header("STRESS TEST SUITE")
    
    print(f"{YELLOW}Warning: Stress tests may take 1-2 hours to complete{RESET}\n")
    
    cmd = ["pytest", "test_stress.py", "-v", "--tb=short", "-m", "slow"]
    return run_command(cmd, "Large-Scale Stress Tests")


def run_module_tests(module):
    """Run tests for specific module"""
    module_map = {
        'robust': 'benchmarks_robust_stats.py',
        'inference': 'benchmarks_inference.py',
        'resampling': 'benchmarks_resampling.py',
        'matrix': 'benchmarks_matrix.py',
    }
    
    if module not in module_map:
        print(f"{RED}Unknown module: {module}{RESET}")
        print(f"Available modules: {', '.join(module_map.keys())}")
        return False
    
    print_header(f"{module.upper()} MODULE TESTS")
    
    cmd = ["pytest", module_map[module], "-v", "--tb=short"]
    return run_command(cmd, f"{module.title()} Module Tests")


def run_all_tests():
    """Run complete test suite"""
    print_header("COMPLETE TEST SUITE")
    print(f"{YELLOW}This will run all tests including stress tests (1-2 hours){RESET}\n")
    
    start = time.perf_counter()
    
    results = []
    results.append(run_fast_tests())
    results.append(run_benchmarks())
    results.append(run_stress_tests())
    
    elapsed = time.perf_counter() - start
    
    print_header("FINAL RESULTS")
    
    if all(results):
        print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED{RESET}")
    else:
        print(f"{RED}{BOLD}✗ SOME TESTS FAILED{RESET}")
    
    print(f"\nTotal time: {elapsed/60:.1f} minutes\n")
    
    return all(results)


def print_usage():
    """Print usage information"""
    print(f"""
{BOLD}bunker-stats Test Runner{RESET}

Usage:
    python run_tests.py <command> [options]

Commands:
    fast          Run fast tests only (< 5 min)
    benchmarks    Run all performance benchmarks (< 30 min)
    safety        Run safety tests (concurrency + memory)
    stress        Run stress tests (1-2 hours)
    all           Run complete test suite (1-2 hours)
    module <name> Run specific module tests

Modules:
    robust        Robust statistics benchmarks
    inference     Inference benchmarks
    resampling    Resampling benchmarks
    matrix        Matrix operations benchmarks

Examples:
    python run_tests.py fast
    python run_tests.py benchmarks
    python run_tests.py module robust
    python run_tests.py all

Additional pytest options can be passed:
    python run_tests.py fast -k "median"
    python run_tests.py benchmarks -v --tb=line
""")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    command = sys.argv[1].lower()
    
    # Map commands to functions
    commands = {
        'fast': run_fast_tests,
        'benchmarks': run_benchmarks,
        'safety': run_safety_tests,
        'stress': run_stress_tests,
        'all': run_all_tests,
    }
    
    if command == 'module':
        if len(sys.argv) < 3:
            print(f"{RED}Error: module name required{RESET}")
            print("Usage: python run_tests.py module <name>")
            return 1
        success = run_module_tests(sys.argv[2].lower())
    elif command in commands:
        success = commands[command]()
    elif command in ['help', '-h', '--help']:
        print_usage()
        return 0
    else:
        print(f"{RED}Unknown command: {command}{RESET}\n")
        print_usage()
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
