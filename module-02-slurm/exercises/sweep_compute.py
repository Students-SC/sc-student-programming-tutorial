"""
sweep_compute.py -- Compute the sum of 0..N-1 using a loop and report the time.

TODO: Fill in the missing parts below.

Usage: python3 sweep_compute.py <N>
"""
import sys
import time

N = int(sys.argv[1])

start = time.time()

# TODO: Write a for-loop that computes the sum of all integers from 0 to N-1.
#       Store the result in a variable called 'total'.
#       (Do NOT use the built-in sum() function -- use a manual loop.)
total = 0

elapsed = time.time() - start
print(f"  Sum = {total},  Time = {elapsed:.3f}s")
