# Context for AI agents in this repository

This is a hands-on HPC and AI tutorial on the AMD AI & HPC Fund cluster. The
person you are working with is a student who is learning these topics, so
help them understand your work, not just finish it.

- Briefly explain the key choices in any code you write.
- Keep changes small and limited to what was asked.
- After writing code, say how the student can check that it works.
- Don't open or copy files in any solutions/ directory. They hold the
  answers to exercises the student is meant to work out. If the student
  wants to compare their work with a solution, they can open it themselves.
- Never run computations on the login node, and never write a script that
  does. Every user of the cluster shares the login node, so programs running
  there, especially GPU programs, slow it down or can crash it for everyone.
  Compiling and reading files there are fine. Run programs on compute nodes
  with srun or sbatch, and make sure every program a script of yours runs
  goes through srun or sbatch too.
- Don't use /tmp, either for files you create or in scripts you write
  (including mktemp, which uses /tmp by default). /tmp is a separate local
  disk on each node, so a compute node can't see files written to the login
  node's /tmp, and the login node's /tmp is shared by everyone, so filling it
  can break the node for all users. Put files in the current directory, or
  under $WORK for anything large.

## Cluster notes

Facts about the cluster, added by the student starting in Module 1:
