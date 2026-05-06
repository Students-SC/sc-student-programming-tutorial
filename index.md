# Hands-On Introduction to HPC and AI

> **SC26 Student Programming Tutorial**
>
> A full-day, hands-on path from logging into an HPC cluster to running parallel
> programs on CPUs and GPUs, then using AI models and agents on the same system.

[Start the Tutorial](README.md) | [Open Module 1](module-01-hpc-foundations/README.md)

| Format | Venue |
|--------|-------|
| Hands-on cluster exercises and challenges | SC26, Chicago · Room TBD |

## Getting Help

If you get stuck, ask for help early. Instructors will be in the room throughout
the tutorial, and we may also provide an online help channel such as Slack. Final
help-channel details will be added before the tutorial.

## Course Path

| Module | Topic | What You'll Do |
|--------|-------|----------------|
| [1](module-01-hpc-foundations/README.md) | HPC Foundations | Explore the cluster, filesystems, software modules, and hardware basics. |
| [2](module-02-slurm/README.md) | Slurm Scheduling | Submit, monitor, and understand jobs running on compute nodes. |
| [3](module-03-openmp/README.md) | OpenMP | Use shared-memory parallelism to speed up CPU programs. |
| [4](module-04-mpi/README.md) | MPI | Coordinate distributed processes and message passing across ranks. |
| [5](module-05-hip/README.md) | HIP GPU Programming | Write GPU kernels and reason about host/device memory movement. |
| [6](module-06-ai-inference-finetuning/README.md) | AI on HPC | Run inference and fine-tune models using GPU-backed batch jobs. |
| [7](module-07-ai-agents/README.md) | AI Agents | Build a tool-using agent and pull the day's HPC skills together. |
| [Reference](commands-reference.md) | Commands Reference | Find common shell, Slurm, OpenMP, MPI, HIP, Python, and agent commands. |

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorial

README
module-01-hpc-foundations/README
module-02-slurm/README
module-03-openmp/README
module-04-mpi/README
module-05-hip/README
module-06-ai-inference-finetuning/README
module-07-ai-agents/README
module-07-ai-agents/exercises/capstone
commands-reference
```
