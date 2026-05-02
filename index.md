# Hands-On Introduction to HPC and AI

<section class="tutorial-hero">
  <p class="eyebrow">SC26 Student Programming Tutorial</p>
  <p class="dek">
    A full-day, hands-on path from logging into an HPC cluster to running parallel
    programs on CPUs and GPUs, then using AI models and agents on the same system.
  </p>
  <div class="hero-actions">
    <a class="button primary" href="README.html">Start the Tutorial</a>
    <a class="button secondary" href="module-01-hpc-foundations/README.html">Open Module 1</a>
  </div>
</section>

<section class="course-strip" aria-label="Course details">
  <div>
    <strong>Format</strong>
    <span>Hands-on cluster exercises and challenges</span>
  </div>
  <div>
    <strong>Venue</strong>
    <span>SC26, Chicago · Room TBD</span>
  </div>
</section>

<section class="help-panel">
  <h2>Getting Help</h2>
  <p>
    If you get stuck, ask for help early. Instructors will be in the room
    throughout the tutorial, and we may also provide an online help channel such
    as Slack. Final help-channel details will be added before the tutorial.
  </p>
</section>

## Course Path

<div class="module-grid">
  <a class="module-card" href="module-01-hpc-foundations/README.html">
    <span class="module-number">01</span>
    <h3>HPC Foundations</h3>
    <p>Explore the cluster, filesystems, software modules, and hardware basics.</p>
  </a>
  <a class="module-card" href="module-02-slurm/README.html">
    <span class="module-number">02</span>
    <h3>Slurm Scheduling</h3>
    <p>Submit, monitor, and understand jobs running on compute nodes.</p>
  </a>
  <a class="module-card" href="module-03-openmp/README.html">
    <span class="module-number">03</span>
    <h3>OpenMP</h3>
    <p>Use shared-memory parallelism to speed up CPU programs.</p>
  </a>
  <a class="module-card" href="module-04-mpi/README.html">
    <span class="module-number">04</span>
    <h3>MPI</h3>
    <p>Coordinate distributed processes and message passing across ranks.</p>
  </a>
  <a class="module-card" href="module-05-hip/README.html">
    <span class="module-number">05</span>
    <h3>HIP GPU Programming</h3>
    <p>Write GPU kernels and reason about host/device memory movement.</p>
  </a>
  <a class="module-card" href="module-06-ai-inference-finetuning/README.html">
    <span class="module-number">06</span>
    <h3>AI on HPC</h3>
    <p>Run inference and fine-tune models using GPU-backed batch jobs.</p>
  </a>
  <a class="module-card" href="module-07-ai-agents/README.html">
    <span class="module-number">07</span>
    <h3>AI Agents</h3>
    <p>Build a tool-using agent and pull the day’s HPC skills together.</p>
  </a>
  <a class="module-card reference-card" href="commands-reference.html">
    <span class="module-number">REF</span>
    <h3>Commands Reference</h3>
    <p>Find common shell, Slurm, OpenMP, MPI, HIP, Python, and agent commands.</p>
  </a>
</div>

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
