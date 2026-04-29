# Module 1 -- HPC Foundations & Cluster Architecture

**Time**: 8:50 -- 9:20 AM (30 min: ~20 min lecture, ~10 min hands-on)

## Learning Objectives

By the end of this module, you will be able to:

- Explain why HPC exists and what problems it solves
- Describe the major components of an HPC cluster
- Navigate the cluster filesystem and use environment modules
- Inspect hardware resources using command-line tools

---

## Key Concepts

### Why HPC?

Some problems are too large or too slow for a single computer:

- **Weather forecasting**: Simulating the atmosphere over a global grid
- **Genomics**: Aligning billions of DNA reads against a reference genome
- **AI training**: Adjusting billions of model parameters over terabytes of data
- **Physics simulations**: Modeling molecular dynamics, fluid flow, or cosmological structure

HPC clusters solve this by combining many computers (called **nodes**) into a single
system and splitting the work across them.

### Anatomy of an HPC Cluster

```
┌─────────────────────────────────────────────────────────────┐
│                        USERS (SSH)                          │
│                            │                                │
│                     ┌──────▼──────┐                         │
│                     │ Login Node  │  ← You are here         │
│                     └──────┬──────┘                         │
│                            │                                │
│               ┌────────────┼────────────┐                   │
│               │     High-Speed Network  │                   │
│               │      (InfiniBand)       │                   │
│               └──┬─────┬─────┬───────┬──┘                   │
│                  │     │     │       │                      │
│              ┌───▼┐ ┌──▼┐ ┌──▼─┐ ┌───▼┐                     │
│              │Node│ │...│ │... │ │Node│  ← Compute nodes    │
│              │ 1  │ │   │ │    │ │ N  │    (CPUs + GPUs)    │
│              └────┘ └───┘ └────┘ └────┘                     │
│                  │     │     │     │                        │
│              ┌───┴─────┴─────┴─────┴───┐                    │
│              │    Shared Storage       │                    │
│              │  (Parallel Filesystem)  │                    │
│              └─────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**Login nodes** are where you:
- Edit code, write scripts, manage files
- Submit jobs to the scheduler
- **Do NOT run compute-intensive programs** (everyone shares these)

**Compute nodes** are where your programs actually run:
- The compute nodes we'll be using in this cluster have 16 CPU cores + 1 AMD MI210 GPU
- You access them by submitting **jobs** through the Slurm scheduler (Module 2)

**High-speed network** (InfiniBand, RoCEv2) connects nodes so they can communicate fast --
critical for MPI programs (Module 4).

**Shared storage** means your files in `$HOME` and `/work1` are visible from every node.
You don't need to copy files to each node.

### Our Cluster: The AUP AI & HPC Cluster

The AUP AI & HPC Cluster has nodes containing multiple generations of AMD Instinct GPUs.
For this tutorial, we'll have access to the following resources:

| Component | Details |
|-----------|---------|
| Login node CPUs | AMD EPYC 7V13 64-Core (2 sockets, 128 cores total) |
| Compute node CPUs | AMD EPYC (16 cores per virtual node) |
| Compute node GPUs | 1x AMD Instinct MI210 (64 GB HBM2e) |
| Compute node RAM | 64 GB |
| GPU software | ROCm 7.2.0 |
| Compiler | GCC 12.2.0 |
| MPI | OpenMPI 4.1.8 |
| Scheduler | Slurm |
| Student partition | `mi2101x` |

### The Software Stack: Environment Modules

HPC clusters use **environment modules** to manage software. Instead of installing
packages globally (like on a laptop), you load and unload modules to make specific
software versions available.

```bash
module list              # What's currently loaded?
module avail             # What's available to load?
module show <name>       # What does a module do?
module load <name>       # Load a module
module unload <name>     # Unload a module
```

On our cluster, the `hpcfund` module is loaded by default and provides the base
environment (GCC, OpenMPI, ROCm, cmake). **Do not run `module purge`** -- it will
remove this base.

### Programming Models (Preview)

| Model | Parallelism Level | Example | Module |
|-------|------------------|---------|--------|
| OpenMP | Threads within a single node | `#pragma omp parallel` | 3 |
| MPI | Processes across multiple nodes | `MPI_Send` / `MPI_Recv` | 4 |
| HIP | GPU threads (thousands) | `kernel<<<blocks, threads>>>()` | 5 |

You'll learn all three today!

---

## Hands-On Exercises (~10 min)

### Core: Explore the Cluster

First, navigate to the exercises directory for this module:

```bash
cd module-01-hpc-foundations/exercises
```

Then run the guided exploration script, which walks you through key commands:

```bash
bash explore_cluster.sh
```

The script will pause between sections so you can read the output. Alternatively,
you can run each command below individually.

#### Step 1: Where Are You?

```bash
hostname                  # What machine are you on?
whoami                    # What's your username?
echo $HOME                # Your home directory
pwd                       # Current working directory
```

#### Step 2: Explore the Filesystem

```bash
ls $HOME                  # Your home directory contents
ls /work1                 # Shared work directory
df -h $HOME               # Disk space available
df -h $WORK
```

#### Step 3: Check the Software Environment

```bash
module list               # Currently loaded modules
module avail              # All available modules
module show hpcfund       # What does the base module provide?
```

#### Step 4: Inspect the Hardware (Login Node)

```bash
lscpu                     # CPU information
lscpu | grep "Model name" # Just the CPU model
lscpu | grep "CPU(s):"    # Number of CPUs
free -h                   # Memory information
```

#### Step 5: Preview the GPU Software

```bash
which hipcc               # Is the HIP compiler available?
hipcc --version           # What version?
rocminfo | head -20       # ROCm system info (first 20 lines)
```

> **Note:** The login node may or may not have GPUs. The compute nodes in the `mi2101x`
> partition definitely do. You'll see the full GPU information when you run jobs in
> Module 2.

#### Step 6: Look at the Cluster

```bash
sinfo                     # Show all partitions and node states
sinfo -p mi2101x          # Just our partition
sinfo -p mi2101x -N -l    # Detailed per-node listing
squeue                    # Current job queue (may be empty or busy)
```

### Challenge

1. **Summarize the cluster.** Based on what you found, write a short paragraph (3-4
   sentences) describing the cluster: how many nodes are in our partition, how many
   CPUs per node, what GPU each node has, and what software is available.

2. **AI agent check.** Copy the output of `rocminfo` and ask your AI coding assistant
   to explain it. Does the agent correctly identify the CPU model? The GPU? Does it
   get anything wrong?

   > **First time using the agent?** Launch the tutorial-provided coding agent
   > (aider) from this directory:
   >
   > ```bash
   > bash ../../setup/launch_aider.sh
   > ```
   >
   > Inside aider, paste the `rocminfo` output and ask for an explanation.
   > Type `/exit` when done. See the top-level README for more on the agent.

---

## Quick Reference

| Command | What It Does |
|---------|-------------|
| `hostname` | Print the name of the machine you're on |
| `lscpu` | Show CPU architecture information |
| `free -h` | Show memory usage in human-readable format |
| `module list` | List currently loaded modules |
| `module avail` | List all available modules |
| `module show <name>` | Show what a module does |
| `sinfo` | Show cluster partition and node information |
| `squeue` | Show the job queue |

---

**Next up:** [Module 2 -- Job Scheduling with Slurm](../module-02-slurm/README.md)
