# Module 2 -- Job Scheduling with Slurm

> **Time:** 9:20 to 10:05 AM CST · 45 min total · ~15 min lecture · ~30 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain why HPC clusters use batch scheduling
- Write a Slurm batch script with appropriate `#SBATCH` directives
- Submit, monitor, and cancel jobs using `sbatch`, `squeue`, and `scancel`
- Run quick commands on compute nodes with `srun`
- Check an agent-written batch script against the cluster's rules

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
| `srun <command>` | Run a command on a compute node | `srun --partition=mi2101x --nodes=1 --time=1:00 hostname` |
| `sinfo` | View partition/node status | `sinfo -p mi2101x` |
| `scontrol show job <id>` | Detailed info about a job | `scontrol show job 12345` |
| `sacct -j <id>` | Accounting info after completion | `sacct -j 12345` |

### Anatomy of a Batch Script

A batch script is a regular shell script with special `#SBATCH` comment lines called
**scheduler directives**. They tell Slurm what resources you need:

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
| `--partition` | Target node group. **Required on this cluster.** Use `mi2101x` for this tutorial. |
| `--nodes` | How many nodes. **Required on this cluster**, and `1` is the most a job can use here. |
| `--ntasks` | Number of MPI ranks (processes). Use `1` unless doing MPI. |
| `--cpus-per-task` | CPU cores per task. Useful for OpenMP threading. |
| `--time` | Maximum run time. **Required on this cluster.** Keep short (5--10 min) to share nodes fairly. |
| `--output` / `--error` | Where stdout/stderr go. `%j` is replaced with the job ID. |

When your job runs, Slurm also sets environment variables that describe it,
such as `SLURM_JOB_ID`, `SLURM_JOB_NODELIST`, and `SLURM_CPUS_ON_NODE`. Scripts
and programs can read them; the programs in Exercises 1 and 2 print several.

### The Job Lifecycle

```
sbatch submit → PENDING (waiting for nodes) → RUNNING (on a node) → COMPLETED
                                                       ↓
                                              output written to file
```

You can also cancel at any stage with `scancel`.

While a job is `PENDING`, the last column of `squeue` (`NODELIST(REASON)`)
says why it is waiting:

| Reason | Meaning |
|--------|---------|
| `Resources` | Waiting for a node to become free |
| `Priority` | Other jobs are ahead of it in the queue |

With 80+ students sharing about 20 nodes, expect to see these today. Waiting
is normal: Slurm starts your job as soon as a node is free.

---

## Hands-On Exercises (~30 min)

First, navigate to the exercises directory for this module:

```bash
cd module-02-slurm/exercises
```

### Exercise 1: Observe the Job Lifecycle

Your first batch script includes a `sleep` so the job stays running long enough
for you to practice monitoring it.

#### Step 1: Look at the Template

```bash
cat first_job.sh
```

#### Step 2: Submit the Job

```bash
sbatch first_job.sh
```

Slurm will print something like `Submitted batch job 12345`. Note the **job ID**.

```{note}
Before that line, you'll see a block of checks from the cluster's
job-submission filter (lines starting with `-->`). That's normal, and you'll
see it for every `sbatch` and `srun`. If the filter rejects a job, the block
says why, for example `runtime limit is required` when a job has no `--time`.
```

#### Step 3: Check the Queue

```bash
squeue -u $USER
```

You should see your job in `PENDING` or `RUNNING` state. Try these while it runs:

```bash
squeue -u $USER                     # Your jobs
scontrol show job <JOBID>           # Detailed job info
squeue -p mi2101x                   # All jobs on our partition
```

#### Step 4: Cancel the Job

Don't wait for it to finish:

```bash
scancel <JOBID>
```

Verify it's gone:

```bash
squeue -u $USER
```

#### Step 5: Let the Job Finish

Submit it again and let it finish. Then read the output:

```bash
sbatch first_job.sh
# Once it starts, it runs for about 50 seconds (it may wait in the queue first)
squeue -u $USER                     # Should disappear when done
cat first-job_<JOBID>.out           # Replace <JOBID> with your job ID
```

```{tip}
`ls -lt *.out | head` shows the most recent output files.
```

`squeue` lists only jobs that are pending or running. To look up jobs that have
finished, including the one you cancelled in Step 4, use `sacct`:

```bash
sacct -X -j <JOBID> --format=JobID,JobName%15,State,Elapsed   # Try both job IDs from this exercise
```

`-X` shows one line per job. `State` shows `COMPLETED` or `CANCELLED`, and
`Elapsed` shows how long the job ran.

---

### Exercise 2: Compile and Run on a Compute Node

Now let's do something more realistic: compile a C program and run it on a
compute node.

#### Step 1: Look at the Source Code

```bash
cat hello_compute.c
```

This program prints information about the compute node, including GPU details.

#### Step 2: Compile on the Login Node

Compiling is lightweight, so it is OK to do on the login node:

```bash
gcc -o hello_compute hello_compute.c
```

#### Step 3: Look at the Batch Script

```bash
cat submit_hello.sh
```

#### Step 4: Submit the Job

```bash
sbatch submit_hello.sh
```

#### Step 5: Examine the Output

Once the job completes, examine its output:

```bash
cat hello-compute_<JOBID>.out
```

**Questions to answer:**
- What is the hostname of the compute node? Is it different from the login node?
- How many CPU cores does the compute node report?
- Does it detect a GPU?

---

### Exercise 3: Interactive Commands with `srun`

`srun` lets you run a single command on a compute node without writing a batch
script. Useful for quick tests.

#### Step 1: Run a Command

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 hostname
```

#### Step 2: Try More Commands

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 lscpu | grep "Model name"
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 rocminfo | grep -E "Marketing Name|Device Type"
```

On the login node, the same `rocminfo` command lists two CPUs and two GPUs.
Your compute node has one of each.

```{note}
`srun` waits for a node, runs the command, and returns. If the partition is
busy, you may wait a moment.
```

```{important}
Every job, even a one-line `srun`, gets a whole compute node to itself until it
finishes. That makes `srun` great for quick commands, but please don't start
an interactive shell on a compute node today (for example, `srun --pty bash`):
an idle shell holds a node that five of your classmates could be using.
```

---

### Exercise 4: Checkpoint -- The GPU Check Job

During Getting Started, you submitted `setup/check_gpu.sh` as a Slurm job. Now
that you've written batch scripts yourself, that job should make more sense.

#### Step 1: Read the Script

```bash
cat ../../setup/check_gpu.sh
```

**Questions:**
- Which partition and time limit does it request?
- Which line chose the name of its output file, and what did `%j` become?

#### Step 2: Run It if You Skipped It

If you didn't run it during Getting Started, or it didn't print
`Status: PASSED`, run it now from this directory:

```bash
sbatch ../../setup/check_gpu.sh
cat check_gpu_<JOBID>.out           # After the job leaves the queue
```

The output file appears in this directory, because a batch job writes its
output where you submitted it. If the check still doesn't pass, ask an
instructor.

---

### Agent Practice: A Batch Script from a Spec

You have written and read batch scripts yourself; now give the agent a
specification and check its script the way the cluster will. This time you
let the agent edit files, but only after you approve each change.

#### Step 1: Give the Agent the Spec

Start Qwen Code in ask-permissions mode, from this directory:

```bash
bash ../../setup/launch_qwen.sh --approval-mode default
```

Ask it:

> Create agent_job.sh: a Slurm batch script that runs ./hello_compute with 8 CPU cores for at most 3 minutes and writes its output to agent-job_<job ID>.out.

The agent may read other files first; watch which ones. When it asks to create
the file, read the script before you approve it.

#### Step 2: Check It Before Slurm Does

Compare the script with the [key directives](#anatomy-of-a-batch-script)
above. Does it name the partition, the node count, and a time limit? Does it
ask for anything this cluster doesn't allow, such as `--gpus`? Predict whether
Slurm will accept it.

#### Step 3: Submit and Iterate

If the agent offers to run `sbatch agent_job.sh`, you can approve that, or
`/quit` and run it yourself. If the filter rejects the job, give the agent the
lines that start with `-->` and ask it to fix the script. When the job
finishes, check its output:

```bash
cat agent-job_<JOBID>.out
```

**Questions:**
- Which directives did the agent get right on its own? Did it copy any from
  the files in this directory?
- What did you, or the filter, have to correct?

Add what you learned to your notes, so the agent gets it right next time:

```bash
echo "- Every job needs --partition, --nodes=1, and --time." >> ~/.qwen/QWEN.md
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

**In practice:** for sweeps over many independent values, HPC users usually
submit a **job array** instead of a loop. Adding `#SBATCH --array=0-3` runs
four copies of the script as separate jobs, which can run on different nodes at
the same time, and each copy reads its own index from `$SLURM_ARRAY_TASK_ID`.
We use a loop today so that 80+ students don't flood the queue.

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
