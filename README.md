# Hands-On Introduction to HPC and AI

> **SC26 Student Programming Tutorial**
>
> - Sunday, November 15, 2026
> - 8:30 AM to 4:30 PM CST
> - McCormick Place, Chicago, IL · [Room W180](https://sc26.conference-program.com/map/?location=rW180)

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

You will receive an RSA token from one of the people distributing tokens. Make sure that you return
the token when you leave.

Once you have the token, ssh to `ssh csepXY@frontier.olcf.ornl.gov` where XY is replaced by the 2-3
digit number assigned to your token. 

### 2. Clone This Repository

```bash
cd ~
git clone https://github.com/Students-SC/sc-student-programming-tutorial.git
cd sc-student-programming-tutorial
```


### 3. Check That Your Jobs Can Use a GPU

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

The tutorial's AI coding agent (Aider) is also preinstalled, so it's ready
to use. See "Using AI Assistants" below.

## Getting Help

If you get stuck, raise your hand. Instructors are in the room throughout the
tutorial to help with cluster access, Slurm jobs, code errors, and conceptual
questions.

## Compute (virtual) Node Details

In this tutorial, a **virtual compute node** is a slice of a larger physical
server with its own CPU cores, memory, and GPU.
[Module 1](module-01-hpc-foundations/README.md#inside-a-compute-node) explains
how this works.

| Resource | Details |
|----------|---------|
| CPU | AMD EPYC 7A53 64-Core Processor |
| GPU | 8x AMD Instinct MI250X (64 GB HBM2e each) per node |
| RAM | 512 GB system DRAM per node |
| Partition | `batch` |
| Scheduler | Slurm |
| Compiler | GCC 12.2, `hipcc` (ROCm 7.2) |
| MPI | Cray MPICH 9.1.0 |

> **Important**
>
> Do not run compute workloads on the login node. All programs must be
> submitted through Slurm (`sbatch` or `srun`). This is both good practice and
> necessary so that all 80+ participants can share the cluster fairly alongside
> the production research and education workloads being run on the system.

## Using AI Assistants

You are encouraged to use AI coding agents throughout the tutorial -- just as
you would in modern development practice. The ground rule:

> **Use the agent, but own the answer.**

Let the agent help you write code, debug errors, and explain concepts -- but
make sure *you* understand what the code does before moving on. Each module
ends with an **Agent Practice** step: after you have done the exercises
yourself, you hand a related task to the agent, predict what should happen,
and check its work with the same tests you used on your own code. Agents
make mistakes; the point is to catch them.

### The tutorial agent: Aider + OLCF Inference Service 

We provide a CLI coding agent for you to use throughout today. It's Aider pointed at the OLCF Inference service. Exercises that say *"ask
your AI agent..."* refer to this tool.

It's preinstalled for everyone. Start it in the directory you are working in:

```bash
cd ~/sc-student-programming-tutorial
bash setup/launch_aider.sh
```

Note that for Aider, you always need to start it from the main `~/sc-student-programming-tutorial`
directory as it relies on being able to read the QWEN.md file there.

Type requests in plain English. Unlike a chatbot, Qwen Code reads and searches
the files in the directory it starts in, and -- when you allow it -- edits files
and runs commands. When it starts anywhere in the tutorial repository, it also
reads `QWEN.md` at the repository's top level: notes for the agent about this
tutorial, which you add to starting in Module 1. It has three modes; press
**Shift+Tab** to switch between them:


Read everything before you approve it. Approve an edit only if you understand
the change, and a command only if you would run it yourself. In particular,
programs belong on compute nodes (`srun` or `sbatch`), never on the login node;
compiling and reading files on the login node are fine.

Inside Aider, it is running by default in 'ask' mode where it will answer questions but 
won't edit files. If you want to switch to 'code' mode, 
type `/code` to switch to that so that your agent can edit files. Type `/quit` to exit the agent.
`/add` followed by the name of files or directories will add those files to the LLMs context
window so the LLM is aware of those files.

> **Note**
>
> If you'd rather use your own coding assistant (Cursor, GitHub Copilot, ChatGPT,
> Claude, etc.), feel free -- the exercises don't require ours. But the
> tutorial-provided one works for everyone with no setup beyond what's already in
> this README.

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
| 8:30 to 8:50 CST | Welcome, Intro & AI Assistants |
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
