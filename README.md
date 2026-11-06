# Hands-On Introduction to HPC and AI

> **SC26 Student Programming Tutorial**
>
> - Sunday, November 15, 2026
> - 8:30 AM to 4:30 PM CST
> - McCormick Place, Chicago, IL

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
| 8 | (Bonus) OpenMP Offload | OpenMP directives for GPUs |

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

### 4. Check That Your Jobs Can Use a GPU

Your instructors have already installed the Python environment for the
afternoon's AI exercises (PyTorch and friends) in a shared location, so there
is nothing for you to install. Instead, run a quick job that checks that you
can use it and a GPU. From the repository root:

```bash
sbatch setup/check_gpu.sh
```

This runs as a Slurm batch job on a compute node. You will learn what `sbatch`
is doing in Module 2; for now, submit the job, wait for it to leave the queue,
and read its output:

```bash
squeue -u $USER                     # Your job disappears when it finishes
cat check_gpu_<JOBID>.out           # Replace <JOBID> with the number sbatch printed
```

You should see `Status: PASSED` and the GPU's name. Module 2 includes a
checkpoint that explains this job and helps anyone who skipped this step catch
up.

### 5. Install the AI Agent

The coding agent (Aider) lives in its own small environment, separate from the
AI/ML venv above. This install is quick and runs right on the login node (no
Slurm job needed):

```bash
bash setup/install_aider.sh
```

You only need to do this once. You'll launch the agent with
`setup/launch_aider.sh` (see "Using AI Assistants" below).

## Getting Help

If you get stuck, ask for help early. Instructors will be in the room throughout
the tutorial and can help with cluster access, Slurm jobs, code errors, and
conceptual questions.

We may also provide an online help channel, such as Slack, for questions during
or after the session. Details will be added here before the tutorial.

## Compute (virtual) Node Details

In this tutorial, a **virtual compute node** is a slice of a larger physical
server with its own CPU cores, memory, and GPU.
[Module 1](module-01-hpc-foundations/README.md#inside-a-compute-node) explains
how this works.

| Resource | Details |
|----------|---------|
| CPU | AMD EPYC 7V13 (16 cores per virtual node) |
| GPU | 1x AMD Instinct MI210 (64 GB HBM2e) per virtual node |
| RAM | 64 GB system DRAM per virtual node |
| Partition | `mi2101x` |
| Scheduler | Slurm |
| Compiler | GCC 12.2, `hipcc` (ROCm 7.2) |
| MPI | OpenMPI 4.1.8 |

```{important}
Do not run compute workloads on the login node. All programs must be
submitted through Slurm (`sbatch` or `srun`). This is both good practice and
necessary so that all 80+ participants can share the cluster fairly alongside
the production research and education workloads being run on the system.
```

## Using AI Assistants

You are encouraged to use AI coding assistants throughout the tutorial -- just
as you would in modern development practice. The ground rule:

> **Use the agent, but own the answer.**

Let the AI help you write code, debug errors, and explain concepts -- but make
sure *you* understand what the code does before moving on. Exercises are
designed so you need to verify and reason about the AI's output.

### The tutorial agent: Aider + a self-hosted coding model

We provide a CLI coding agent for you to use throughout the day -- no accounts,
no API keys, no cost. It's [Aider](https://aider.chat) (a popular open-source
coding agent) pointed at **Qwen3-Coder-30B-A3B-Instruct**, served on a
dedicated MI300X compute node by your instructors. Exercises that say *"ask
your AI agent..."* refer to this tool.

Launch it from inside the repo after completing the setup steps above:


```bash
# In a new terminal, navigate to the tutorial directory
cd ~/sc-student-programming-tutorial


# Launch aider (uses the agent you installed during Getting Started, pointed at the local model)
bash ~/sc-student-programming-tutorial/setup/launch_aider.sh
```

Inside Aider, type questions in plain English. By default the agent runs in
**ask mode**: it reads the files you give it and explains what to do, but it
does not edit them -- understanding and writing the code is your job. After starting 
Aider, add files or directories to your context with the `/read` command (e.g. run
`/read module-03-slurm` to get Aider to read its contents so it can answer questions
about it). Type
`/help` to see commands, `/exit` to quit.

```{note}
If you'd rather use your own coding assistant (Cursor, GitHub Copilot, ChatGPT,
Claude, etc.), feel free -- the exercises don't require ours. But the
tutorial-provided one works for everyone with no setup beyond what's already in
this README.
```

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
module-08-openmp-offload/           OpenMP for GPU programming
```

Each module directory contains a `README.md` with instructions and an `exercises/`
folder with starter code and templates.

## Schedule

| Time | Session |
|------|---------|
| 8:30 to 8:50 CST | Welcome & AI Assistants Intro |
| 8:50 to 9:20 CST | Module 1: HPC Foundations |
| 9:20 to 10:05 CST | Module 2: Slurm |
| 10:05 to 10:20 CST | *Break* |
| 10:20 to 11:10 CST | Module 3: OpenMP |
| 11:10 to 12:00 CST | Module 4: MPI |
| 12:00 to 12:45 CST | *Working Lunch* |
| 12:45 to 1:45 CST | Module 5: HIP GPU Programming |
| 1:45 to 2:00 CST | *Break* |
| 2:00 to 3:00 CST | Module 6: AI Inference & Fine-Tuning |
| 3:00 to 3:50 CST | Module 7: AI Agents & Capstone |
| 3:50 to 4:10 CST | Wrap-Up |
