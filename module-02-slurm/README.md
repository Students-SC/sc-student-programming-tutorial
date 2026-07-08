# Module 2 -- Job Scheduling with Slurm

> **Time:** 9:20 to 10:05 AM CST · 45 min total · ~15 min lecture · ~30 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain why HPC clusters use batch scheduling
- Write a Slurm batch script with appropriate `#SBATCH` directives
- Submit, monitor, and cancel jobs using `sbatch`, `squeue`, and `scancel`
- Run quick commands on compute nodes with `srun`

---

## Key Concepts

### Why Batch Scheduling?

Imagine 80+ students all trying to run programs on 20 compute nodes at the same
time. Without coordination, some nodes would be overloaded while others sit idle,
and everyone's programs would interfere with each other.

A **batch scheduler** solves this by:

1. Accepting job requests and placing them in a **queue**
2. Allocating compute nodes to jobs in a fair order
3. Running each job in isolation on its assigned node(s)
4. Releasing the node when the job finishes

Our cluster uses **Slurm** (Simple Linux Utility for Resource Management), the
most widely used scheduler in HPC.

### Slurm Commands at a Glance

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch script.sh` | Submit a batch job | `sbatch my_job.sh` |
| `squeue` | View the job queue | `squeue -u $USER` |
| `scancel <jobid>` | Cancel a job | `scancel 12345` |
| `srun <command>` | Run a command on a compute node | `srun hostname` |
| `sinfo` | View partition/node status | `sinfo -p mi2101x` |
| `scontrol show job <id>` | Detailed info about a job | `scontrol show job 12345` |
| `sacct -j <id>` | Accounting info after completion | `sacct -j 12345` |

### Anatomy of a Batch Script

A batch script is a regular shell script with special `#SBATCH` comment lines that
tell Slurm what resources you need:

```bash
#!/bin/bash
#SBATCH --job-name=my-job          # Name shown in squeue
#SBATCH --partition=mi2101x        # Which partition (group of nodes)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --time=5:00                # Max wall time (MM:SS or HH:MM:SS)
#SBATCH --output=my-job_%j.out     # Stdout file (%j = job ID)
#SBATCH --error=my-job_%j.err      # Stderr file

# --- Your program runs below this line ---
echo "Hello from $(hostname) at $(date)"
```

**Key directives:**

| Directive | Meaning |
|-----------|---------|
| `--partition` | Target node group. Use `mi2101x` for this tutorial. |
| `--nodes` | How many nodes. Use `1` for most exercises. |
| `--ntasks` | Number of MPI ranks (processes). Use `1` unless doing MPI. |
| `--cpus-per-task` | CPU cores per task. Useful for OpenMP threading. |
| `--time` | Maximum run time. Keep short (5--10 min) to share nodes fairly. |
| `--output` / `--error` | Where stdout/stderr go. `%j` is replaced with the job ID. |

### The Job Lifecycle

```
sbatch submit → PENDING (waiting for nodes) → RUNNING (on a node) → COMPLETED
                                                       ↓
                                              output written to file
```

You can also cancel at any stage with `scancel`.

---

## Hands-On Exercises (~30 min)

First, navigate to the exercises directory for this module:

```bash
cd module-02-slurm/exercises
```

### Exercise 1: Observe the Job Lifecycle (Core)

Your first batch script includes a `sleep` so the job stays running long enough
for you to practice monitoring it.

**Step 1:** Look at the template:

```bash
cat first_job.sh
```

**Step 2:** Submit it:

```bash
sbatch first_job.sh
```

Slurm will print something like `Submitted batch job 12345`. Note the **job ID**.

**Step 3:** Immediately check the queue:

```bash
squeue -u $USER
```

You should see your job in `PENDING` or `RUNNING` state. Try these while it runs:

```bash
squeue -u $USER                     # Your jobs
scontrol show job <JOBID>           # Detailed job info
squeue -p mi2101x                   # All jobs on our partition
```

**Step 4:** Cancel the job (don't wait for it to finish):

```bash
scancel <JOBID>
```

Verify it's gone:

```bash
squeue -u $USER
```

**Step 5:** Submit it again and let it finish. Then read the output:

```bash
sbatch first_job.sh
# Wait ~70 seconds for it to complete...
squeue -u $USER                     # Should disappear when done
cat first-job_<JOBID>.out           # Replace <JOBID> with your job ID
```

```{tip}
`ls -lt *.out | head` shows the most recent output files.
```

---

### Exercise 2: Compile and Run on a Compute Node (Core)

Now let's do something more realistic: compile a C program and run it on a
compute node.

**Step 1:** Look at the source code:

```bash
cat hello_compute.c
```

This program prints information about the compute node, including GPU details.

**Step 2:** Compile it on the login node (compiling is lightweight, that's OK):

```bash
gcc -o hello_compute hello_compute.c
```

**Step 3:** Look at the batch script that runs it:

```bash
cat submit_hello.sh
```

**Step 4:** Submit:

```bash
sbatch submit_hello.sh
```

**Step 5:** Once it completes, examine the output:

```bash
cat hello-compute_<JOBID>.out
```

**Questions to answer:**
- What is the hostname of the compute node? Is it different from the login node?
- How many CPU cores does the compute node report?
- Does it detect a GPU?

---

### Exercise 3: Interactive Commands with `srun` (Core)

`srun` lets you run a single command on a compute node without writing a batch
script. Useful for quick tests.

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 hostname
```

Try a few more:

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 lscpu | grep "Model name"
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 rocminfo | head -30
```

> [!note]
> `srun` waits for a node, runs the command, and returns. If the partition is
> busy, you may wait a moment.

---

### Exercise 4: Python Environment Checkpoint (Core)

In Getting Started Step 4, you submitted `setup/setup_venv.sh` as a Slurm job.
Now that you've seen `sbatch`, `squeue`, and output files, that command should
make more sense: it requested a compute node and installed the Python packages
used by the tutorial agent and the afternoon AI modules.

First, verify that the venv exists:

```bash
source "$WORK/sc26_venv/bin/activate"
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
deactivate
```

If that works, you're done. If you skipped Getting Started Step 4 or the venv is
missing, create it now from this module's `exercises` directory:

```bash
sbatch ../../setup/setup_venv.sh
```

This takes a few minutes. Check progress with:

```bash
squeue -u $USER
tail -f setup_venv_<JOBID>.out      # Watch the log (Ctrl+C to stop)
```

Once the job finishes, run the verification commands again:

```bash
source "$WORK/sc26_venv/bin/activate"
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
deactivate
```

---

### Challenge Exercises

#### Challenge A: Parameter Sweep

Write a batch script that runs a "parameter sweep" -- the same computation with
different input values. Two template files are provided:

```bash
cat parameter_sweep.sh     # The batch script (shell loop)
cat sweep_compute.py        # The Python script (TODO: add the loop)
```

Fill in the TODO sections in **both** files and submit the batch script. You
should see timing results for each value of N, showing how compute time grows.

#### Challenge B: Explore Resource Requests

Experiment with different Slurm directives and observe their effect:

```bash
# Request multiple tasks (for MPI later)
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 hostname

# Request multiple CPUs per task (for OpenMP later)
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=8 \
  bash -c 'echo "I have $SLURM_CPUS_PER_TASK CPUs"'

# Check the GPU on the node (every mi2101x node has an MI210)
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 \
  bash -c 'echo "GPU node: $(hostname)"; rocm-smi --showproductname 2>/dev/null | head -10'
```

**Question:** What happens if you request `--ntasks=4` with `--nodes=1`? How many
times does `hostname` print?

---

## Quick Reference

| I want to... | Command |
|--------------|---------|
| Submit a job | `sbatch script.sh` |
| See my jobs | `squeue -u $USER` |
| See all jobs on our partition | `squeue -p mi2101x` |
| Get details on a job | `scontrol show job <JOBID>` |
| Cancel a job | `scancel <JOBID>` |
| Cancel all my jobs | `scancel -u $USER` |
| Run a quick command on a node | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 <cmd>` |
| See node availability | `sinfo -p mi2101x` |

---

**Next up:** [Module 3 -- Shared-Memory Parallelism with OpenMP](../module-03-openmp/README.md)
