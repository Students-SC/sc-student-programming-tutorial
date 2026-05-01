# Hands-On Introduction to HPC and AI

**SC26 Student Programming Tutorial**
Sunday, November 15, 2026 | 8:30 AM -- 4:30 PM CST
McCormick Place, Chicago, IL

---

## What You'll Learn

This full-day, hands-on tutorial takes you from "What is a supercomputer?" to writing
parallel programs that run on GPUs and training AI models -- all on a real HPC cluster.

| Module | Topic | Key Skills |
|--------|-------|------------|
| 1 | HPC Foundations | Cluster architecture, environment modules |
| 2 | Job Scheduling with Slurm | Batch scripts, `sbatch`, `squeue`, `scancel` |
| 3 | Shared-Memory Parallelism | OpenMP directives, threading |
| 4 | Distributed-Memory Parallelism | MPI communication, multi-process programs |
| 5 | GPU Programming | HIP kernels, GPU memory management |
| 6 | AI on HPC | Inference, fine-tuning with LoRA |
| 7 | AI Agents | Building a tool-using agent, capstone challenge |

## Prerequisites

- Basic familiarity with a Linux command line (navigating directories, editing files)
- Some programming experience in any language (C, Python, etc.)
- No HPC or parallel programming experience required

## Getting Started

### 1. Connect to the Cluster

You will receive login credentials for the
[AMD University Program (AUP) AI & HPC Cluster](https://amdresearch.github.io/hpcfund/index.html).
Connect via SSH:

```bash
ssh <your-username>@hpcfund.amd.com
```

### 2. Clone This Repository

```bash
cd ~
git clone https://github.com/Students-SC/sc-student-programming-tutorial.git
cd sc-student-programming-tutorial
```

### 3. Verify Your Environment

Run the environment check script to confirm everything is set up:

```bash
bash setup/check_environment.sh
```

You should see [PASS] for all items. If anything fails, ask an instructor for help.

### 4. Set Up Your Python Environment

Create the Python virtual environment now, before starting the modules. You will
use this same environment several times throughout the day, including for the
tutorial-provided AI agent and the afternoon AI exercises.

From the repository root:

```bash
sbatch setup/setup_venv.sh
```

This runs as a Slurm batch job, so it may take a few minutes. You will learn what
`sbatch` is doing in Module 2; for now, submit the job and wait for it to finish:

```bash
squeue -u $USER
tail -f setup_venv_<JOBID>.out      # Watch the log (Ctrl+C to stop)
```

When the job completes, verify the environment:

```bash
source "$WORK/sc26_venv/bin/activate"
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
deactivate
```

You only need to create this venv once. Module 2 includes a checkpoint that
explains the Slurm command you used here and helps anyone who skipped this step
catch up.

## Compute (virtual) Node Details

| Resource | Details |
|----------|---------|
| CPU | AMD EPYC (16 cores per virtual node) |
| GPU | 1x AMD Instinct MI210 (64 GB HBM2e) per node |
| RAM | 64 GB system DRAM per node |
| Partition | `mi2101x` |
| Scheduler | Slurm |
| Compiler | GCC 12.2, `hipcc` (ROCm 7.2) |
| MPI | OpenMPI 4.1.8 |

> **Important:** Do not run compute workloads on the login node. All programs must be
> submitted through Slurm (`sbatch` or `srun`). This is both good practice and necessary
> so that all 80+ participants can share the cluster fairly alongside the production
> research and education workloads being run on the system.

## Using AI Assistants

You are encouraged to use AI coding assistants throughout the tutorial -- just
as you would in modern development practice. The ground rule:

> **Use the agent, but own the answer.**

Let the AI help you write code, debug errors, and explain concepts -- but make
sure *you* understand what the code does before moving on. Exercises are
designed so you need to verify and reason about the AI's output.

### The tutorial agent: aider + a self-hosted coding model

We provide a CLI coding agent for you to use throughout the day -- no accounts,
no API keys, no cost. It's [aider](https://aider.chat) (a popular open-source
coding agent) pointed at **Qwen3-Coder-30B-A3B-Instruct**, served on a
dedicated MI300X compute node by your instructors. Exercises that say *"ask
your AI agent..."* refer to this tool.

Launch it from inside the repo after completing the setup steps above:

```bash
# cd into a directory you want aider to work in (your code, exercises, etc.)
cd module-03-openmp/exercises

# Launch aider (this activates the venv from Step 4 and points at the local model)
bash ~/sc-student-programming-tutorial/setup/launch_aider.sh
```

Inside aider, type instructions in plain English. Aider will read and edit
files in the current directory. Type `/help` to see commands, `/exit` to quit.

> **Note for advanced students:** If you'd rather use your own coding assistant
> (Cursor, GitHub Copilot, ChatGPT, Claude, etc.), feel free -- the exercises
> don't require ours. But the tutorial-provided one works for everyone with no
> setup beyond what's already in this README.

### How the agent backend works (preview of Module 7)

The agent is **the same kind of system you'll build yourself in Module 7**:
an LLM running on the cluster, served via an OpenAI-compatible API, queried
in a reasoning loop. By the end of the day you'll understand exactly how it
works under the hood.

## Repository Layout

```
setup/                              Environment check & Python venv setup
module-01-hpc-foundations/          Cluster architecture & environment
module-02-slurm/                    Batch scheduling with Slurm
module-03-openmp/                   Shared-memory parallelism
module-04-mpi/                      Distributed-memory parallelism
module-05-hip/                      GPU programming with HIP
module-06-ai-inference-finetuning/  AI inference & fine-tuning
module-07-ai-agents/                AI agents & capstone challenge
```

Each module directory contains a `README.md` with instructions and an `exercises/`
folder with starter code and templates.

## Schedule

| Time | Session |
|------|---------|
| 8:30 -- 8:50 | Welcome & AI Assistants Intro |
| 8:50 -- 9:20 | Module 1: HPC Foundations |
| 9:20 -- 10:05 | Module 2: Slurm |
| 10:05 -- 10:20 | *Break* |
| 10:20 -- 11:10 | Module 3: OpenMP |
| 11:10 -- 12:00 | Module 4: MPI |
| 12:00 -- 12:45 | *Working Lunch* |
| 12:45 -- 1:45 | Module 5: HIP GPU Programming |
| 1:45 -- 2:00 | *Break* |
| 2:00 -- 3:00 | Module 6: AI Inference & Fine-Tuning |
| 3:00 -- 3:50 | Module 7: AI Agents & Capstone |
| 3:50 -- 4:10 | Wrap-Up |
