# Module 1 -- HPC Foundations & Cluster Architecture

> **Time:** 8:50 to 9:20 AM CST · 30 min total · ~20 min lecture · ~10 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain why HPC exists and what problems it solves
- Explain why performance now comes from parallelism, and what limits the
  speedup you can get
- Describe the major components of an HPC cluster
- Navigate the cluster filesystem and use environment modules
- Inspect hardware resources using command-line tools
- Give an AI agent the facts about your cluster that it doesn't know on its own

---

## Key Concepts

### Why HPC?

Some problems are too large or too slow for a single computer:

- **Weather forecasting**: Simulating the atmosphere over a global grid
- **Genomics**: Aligning billions of DNA reads against a reference genome
- **AI training**: Adjusting billions of model parameters over terabytes of data
- **Physics simulations**: Modeling molecular dynamics, fluid flow, or cosmological structure

Why not just build a faster computer? For decades, processors got faster mainly
by raising their clock speed. Around the mid-2000s that stopped: faster clocks
made chips draw too much power and run too hot, so clock speeds have stayed at a
few gigahertz ever since. Chips kept gaining transistors, though, so performance
now comes from **parallelism**: more cores per CPU, GPUs with thousands of
simpler cores, and many computers working together. Each programming model
you'll learn today is a way to use one of these kinds of parallelism.

HPC clusters solve this by combining many computers (called **nodes**) into a single
system and splitting the work across them.

You're at the right conference to see this at scale. The
[TOP500](https://www.top500.org) list of the world's fastest supercomputers is
published twice a year, and the November edition is announced here at SC.
Several recent leaders are built on AMD hardware: Frontier pairs AMD EPYC CPUs
with AMD Instinct MI250X GPUs, and El Capitan uses AMD Instinct MI300A chips
that combine CPU and GPU cores. Today you'll program Frontier's MI250X
GPUs, which use AMD's CDNA 2 architecture.

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
│               └──┬─────┬─────┬───────┬──┘                   │
│                  │     │     │       │                      │
│              ┌───▼┐ ┌──▼┐ ┌──▼─┐ ┌───▼┐                     │
│              │Node│ │...│ │... │ │Node│  ← Compute nodes    │
│              │ 1  │ │   │ │    │ │ N  │    (CPUs + GPUs)    │
│              └────┘ └───┘ └────┘ └────┘                     │
│                  │     │     │     │                        │
│       ┌──────────┴─────┴─────┴─────┴────────────────┐       │
│       │     Shared storage visible to all nodes     │       │
│       ├──────────────────────┬──────────────────────┤       │
│       │ $HOME                │ $WORK (Orion)        │       │
│       │ Home filesystem      │ Parallel filesystem  │       │
│       └──────────────────────┴──────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Login nodes** are where you:
- Edit code, write scripts, manage files
- Submit jobs to the scheduler
- **Do NOT run compute-intensive programs** (everyone shares these)

**Compute nodes** are where your programs actually run:
- Each Frontier compute node has 64 CPU cores and four MI250X packages, exposed as eight GPU devices
- You access them by submitting **jobs** through the Slurm scheduler (Module 2)

**High-speed network** connects nodes so they can communicate fast --
critical for HPC and AI applications.

**Shared storage** means your files are visible from every node, so you don't need
to copy them to each node. `$HOME` is your smaller home filesystem, while `$WORK`
points to the tutorial work area on the Orion parallel filesystem (`/lustre/orion`).

### Inside a Compute Node

A compute node is a complete computer, built for throughput rather than for a
desk:

- **CPUs** in one or more **sockets**, each CPU with many **cores** (a Frontier
  compute node has one 64-core AMD EPYC CPU)
- **Memory (DRAM)** shared by all the cores in the node
- **GPUs**, each with its own fast memory (**HBM**, high-bandwidth memory). Data
  has to be copied between CPU memory and GPU memory, as you'll see in Module 5.
- A **network card** that connects the node to the high-speed network

**Frontier compute nodes are physical nodes.** Each has one CPU and four
MI250X packages. Each package contains two graphics compute dies (GCDs), so
Slurm and HIP expose eight GPU devices per node. Each device has its own
64 GB of HBM2e memory; the exercises use one device unless stated otherwise.

### Our Cluster: OLCF Frontier

The tutorial uses Frontier at the Oak Ridge Leadership Computing Facility.

| Component | Details |
|-----------|---------|
| Compute node CPU | AMD EPYC 7A53, 64 physical cores |
| Compute node GPUs | 4x AMD Instinct MI250X packages, exposed as 8 GPU devices |
| GPU memory | 64 GB HBM2e per device |
| Compute node RAM | 512 GB |
| GPU software | ROCm 7.2 |
| Compiler | GCC 12.2, `hipcc` |
| MPI | Cray MPICH 9.1.0 |
| Scheduler | Slurm |
| Student partition | `batch` |
| Tutorial account | `TRN001` |
| Tutorial reservation | `sc26studentprogram` |

See the [OLCF Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
for the node architecture. The exercises often use only 16 CPU cores; this
is an exercise size, not the total number of cores on a node.

### The Software Stack: Environment Modules

HPC clusters often use **environment modules** to manage software. Instead of installing
packages globally (like on a laptop), you load and unload modules to make specific
software versions available.

```bash
module list              # What's currently loaded?
module avail             # What's available to load?
module show <name>       # What does a module do?
module load <name>       # Load a module
module unload <name>     # Unload a module
```

On Frontier, inspect the loaded programming environment with `module list`.
The GPU exercises use `module load rocm/7.2`; the Python exercises also use
`module load miniforge3`. Keep the instructor-provided environment loaded,
and ask an instructor before changing the programming environment.

### Programming Models (Preview)

| Model | Parallelism Level | Example | Module |
|-------|------------------|---------|--------|
| OpenMP | Threads within a single node | `#pragma omp parallel` | 3 |
| MPI | Processes on one or many nodes (today: one node) | `MPI_Send` / `MPI_Recv` | 4 |
| HIP | GPU threads (thousands) | `kernel<<<blocks, threads>>>()` | 5 |

You'll learn all three today!

### How Much Faster? Amdahl's Law

Parallelism speeds up only the part of a program that can run in parallel. If a
fraction *p* of the run time can be split across *N* cores and the rest stays
serial, the best possible speedup is:

```text
                    1
speedup(N) = ---------------
             (1 - p) + p / N
```

For example, if 95% of a program parallelizes perfectly (*p* = 0.95), 16 cores
give at most 1 / (0.05 + 0.95 / 16) ≈ **9.1x**, not 16x. Even unlimited cores
can't beat 1 / 0.05 = **20x**: the serial 5% becomes the bottleneck.

You'll measure real speedups in Module 3 (OpenMP threads) and Module 4 (MPI
ranks). When they fall short of perfect, Amdahl's law is one reason; overheads
such as starting threads and passing messages are others.

---

## Hands-On Exercises (~10 min)

### Exercise 1: Explore the Cluster

This module has no files to work with, so you can run these commands from any
directory. Work through the steps below, typing each command yourself and
reading the output before moving on.


#### Step 1: Where Are You?

```bash
hostname          # Print the name of the machine you are logged into.
whoami            # Print your username on this system.
echo $HOME        # Print the path to your home directory (the $HOME variable).
pwd               # Print the current working directory.
date              # Print the current date and time.
```

#### Step 2: Explore the Filesystem

```bash
ls -la $HOME      # List everything in your home directory, long format, including hidden files.
df -h $HOME       # Show disk space usage for your home directory.
```

#### Step 3: Check the Software Environment

```bash
module list           # List the environment modules currently loaded in your shell.
module avail          # Lists the modules that can be loaded given the currently loaded modules
module spider         # Lists all the available modules on the system
module show rocm/7.2  # Show details about the ROCm module used in this tutorial.
```

#### Step 4: Inspect the Hardware (Login Node)

```bash
lscpu             # Print CPU architecture details: model, sockets, cores, threads.
free -h           # Show memory usage and total RAM in human-readable units.
```

> **Note**
>
> This is the LOGIN node. Frontier compute nodes have 64 CPU cores and 512 GB RAM.

#### Step 5: Preview the GPU Software

```bash
which hipcc           # Find the path of the HIP compiler (AMD's equivalent of nvcc).
hipcc --version       # Print the HIP compiler version.
srun -A TRN001 --reservation=sc26studentprogram --partition=batch --nodes=1 --time=2:00 --ntasks=1 rocminfo | grep -E "Marketing Name|Device Type"
```

`rocminfo` prints over 300 lines, so `grep` keeps just the lines that name each
device.

> **Note**
>
> Login nodes are shared by everyone. Inspecting the compiler is fine here,
> but run GPU discovery and GPU programs on a compute node through Slurm.
> Starting in Module 2, jobs use the MI250X devices on Frontier compute nodes.

#### Step 6: Look at the Cluster

```bash
sinfo                       # Show all Slurm partitions and the state of their nodes.
sinfo -p batch -N -l      # Show node-level details just for the batch partition (the one we'll use).
squeue -p batch           # Show the job queue for our partition (plain 'squeue' lists every job on the cluster).
```

> **Note**
>
> `sinfo` lists every partition on the cluster, but this tutorial uses only
> `batch`, with account `TRN001` and reservation `sc26studentprogram`.
> Include these options, a node count, and a time limit in tutorial jobs. Nodes
> set aside for this tutorial show the state `resv` (reserved) while they are
> idle.

### Agent Practice: Teach Your Agent the Cluster

An AI agent knows only what it was trained on and what it can read. In this
step you see what that means, and you start adding notes to the file that
the tutorial's agent, Qwen Code, reads whenever it starts in the tutorial
repository. You will add to it all day.

#### Step 1: Ask Without Providing the Agent Additional Information

Start Qwen Code in an empty directory outside the repository, so it has no
files to read:

```bash
mkdir -p ~/agent-scratch && cd ~/agent-scratch
bash ~/sc-student-programming-tutorial/setup/launch_qwen.sh
```

Ask it:

> Give me the srun command to run hostname on a compute node of this tutorial's partition.

Note what it says, then type `/quit`.

#### Step 2: Provide Agent with Additional Information

The tutorial repository has a file for exactly this: `QWEN.md`, at its top
level. Qwen Code reads it whenever it starts anywhere inside the repository.
Look at what it already says:

```bash
cd ~/sc-student-programming-tutorial
cat QWEN.md
```

Add facts from Exercise 1 under its "Cluster notes" heading:

```bash
cat >> QWEN.md << 'EOF'
- This tutorial uses Frontier: batch partition, account TRN001, reservation sc26studentprogram; 64 CPU cores, 512 GB DRAM, and 8 MI250X GPU devices per node (64 GB HBM2e per device).
- AMD GPUs: use rocm-smi and rocminfo, never nvidia-smi.
EOF
```

Then add one fact of your own from Exercise 1, for example:

```bash
echo "- Large files go in \$WORK, not \$HOME." >> QWEN.md
```

Look at what the configuration file says now:

```bash
cat QWEN.md
```

#### Step 3: Ask Again, Then Let It Try

Start Qwen Code from the top of the repository, where it reads `QWEN.md`, and
ask the same question:

```bash
bash setup/launch_qwen.sh
```

Compare its answer with Step 1. Then ask it to run the command:

> Run that command.

Qwen Code asks before running anything. Approve the `srun` command: `srun`
asks Slurm for a compute node, so it's fine to start from the login node.

Expect the cluster to reject the agent's first attempts since there are required
Slurm options the agent doesn't know about yet. Each time, watch the agent read
the rejection and fix the command, then approve its next try. (If its first try
already works, ask it which options it added and why.)

When the command succeeds, type `/quit` and add the facts the agent discovered
to `QWEN.md`, one line each:

```bash
echo "- <first fact>" >> QWEN.md
echo "- <second fact>" >> QWEN.md
...
```

#### Step 4: Ask Again With the Required Information

Now start the agent again and ask it the same question.

```bash
bash setup/launch_qwen.sh
```

**Questions:**
- What changed in the answers after you added the additional information from
  steps 3 and 4, and why?
- What time limit did the agent choose? Is that a sensible request for
  `hostname` on a shared cluster?

> **Tip**
>
> Teams that use coding agents keep a file like this in every project (often
> called `AGENTS.md`; Qwen Code reads `QWEN.md`), so every session starts from
> the same facts.

### Challenge

1. **Summarize the cluster.** Based on what you found, write a short paragraph (3-4
   sentences) describing the cluster: how many nodes are in our partition, how many
   CPUs per node, what GPU each node has, and what software is available.

2. **AI agent check.** Save the output from Exercise 1, Step 5, plus each device's number of
   compute units, to a file the agent can read:

   ```bash
   rocminfo | grep -E "Marketing Name|Device Type|Compute Unit" > ~/agent-scratch/rocminfo.txt
   ```

   Start Qwen Code in `~/agent-scratch` as in Agent Practice and ask it to
   explain each line of `@rocminfo.txt` (typing `@` lets you pick a file in
   the current directory).

   ```bash
   cd ~/agent-scratch
   bash ../sc-student-programming-tutorial/setup/launch_qwen.sh
   ```

   Does the agent correctly identify the CPU model? The
   GPU? What does it say a "compute unit" is for the CPU, and for the GPU?
   Check its answer against the cluster table above. Does it get anything
   wrong? See [Using AI Assistants](../README.md#using-ai-assistants) for more
   on the agent.

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
| `squeue -p batch` | Show the job queue for our partition |

---

**Next up:** [Module 2 -- Job Scheduling with Slurm](../module-02-slurm/README.md)
