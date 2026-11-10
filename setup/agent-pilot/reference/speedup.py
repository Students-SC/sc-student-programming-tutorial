import re
import sys

times = {}
threads = None
with open(sys.argv[1]) as f:
    for line in f:
        m = re.match(r"=== OMP_NUM_THREADS=(\d+) ===", line)
        if m:
            threads = int(m.group(1))
        m = re.match(r"Time\s+=\s+([\d.]+) seconds", line)
        if m and threads is not None:
            times[threads] = float(m.group(1))

base = times[1]
print(f"{'Threads':>8} {'Time (s)':>10} {'Speedup':>8}")
for n in sorted(times):
    print(f"{n:>8} {times[n]:>10.3f} {base / times[n]:>8.2f}")
