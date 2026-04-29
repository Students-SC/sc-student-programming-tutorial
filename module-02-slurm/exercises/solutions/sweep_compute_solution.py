"""
sweep_compute_solution.py -- Solution for the parameter sweep Python script.

Usage: python3 sweep_compute_solution.py <N>
"""
import sys
import time

N = int(sys.argv[1])

start = time.time()

total = 0
for i in range(N):
    total += i

elapsed = time.time() - start
print(f"  Sum = {total},  Time = {elapsed:.3f}s")
