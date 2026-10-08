# Challenges

This section is the home for optional challenges that extend the tutorial
modules.

Challenge activities will be added here as module owners review their sections
of the curriculum.

## Module 3 - OpenMP Challenges

The files for these challenges are located in `module-03-openmp-challenges` directory.

### Challenge A: Find the Race Condition

The file `pi_race.c` has a **deliberate bug** -- a
race condition. Compile and run it:

```bash
gcc -fopenmp -O2 -o pi_race pi_race.c -lm
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=16 \
  bash -c 'export OMP_NUM_THREADS=8; ./pi_race'
```

Run it several times. Notice the answer changes each time! Can you spot and fix
the bug? (Hint: compare it to your working `pi_openmp.c`.)

Try asking your AI agent: *"This OpenMP code gives wrong answers. Can you find
the race condition?"* Does it identify the problem correctly?

### Challenge B: Matrix-Vector Multiply

Parallelize a matrix-vector multiplication using OpenMP. A template is at:

```bash
cat matvec_openmp.c
```

The outer loop over rows is embarrassingly parallel -- each row of the output
can be computed independently. Add the appropriate OpenMP directive and compare
performance with the serial version.



### Challenge C: Exploring other clauses for `pragma omp for`

If you look at the [OpenMP API spec for `pragma for`](https://www.openmp.org/spec-html/5.0/openmpsu41.html#x64-1290002.9.2), you will see it has a number of optional clauses you can use. You've seen one already: `reduction`. Try exploring some of the others. For example:

1. Could you apply the `collapse` clause for the `matvec_openmp.c` exercise? Does that still produce the correct output? If not, why?
2. Could you apply the `schedule` clause for the `pi_race.c` and for `matvec_openmp.c`? Is there any difference in execution time between using `schedule(static)` and `schedule(dynamic)` for `omp parallel for`?
