# vLLM Inference Server -- Instructor Deployment Guide

This directory contains scripts to deploy a shared vLLM inference server for
the tutorial's Qwen Code agents. The server runs inside the official
`vllm/vllm-openai-rocm` Docker image (via Apptainer/Singularity) on a dedicated
MI300X compute node, providing an OpenAI-compatible API that all students can
query concurrently.

## Architecture

```
  ┌───────────────┐         ┌───────────────────────────────────┐
  │  Login Node   │         │  Dedicated mi3001x Compute Node   │
  │               │         │                                   │
  │  Student      │  HTTP   │  Apptainer Container              │
  │  Qwen Code    │────────►│  vLLM + ROCm + PyTorch            │
  │  clients      │         │  Qwen3-Coder-30B-A3B-Instruct     │
  │               │         │  Port 8321                        │
  └───────────────┘         │  1x AMD Instinct MI300X (192 GB)  │
                            └───────────────────────────────────┘
```

The login and compute nodes are on the same internal network, so HTTP requests
to the server work directly. The model is a 30B-parameter Mixture-of-Experts
coder model with only ~3B parameters active per token, so inference is fast
even under heavy concurrent load.

## Shared Tutorial Directory

All resources that need to be visible to multiple people live under a single
shared directory in the student project's space. The agent server's files --
the container SIF, the model cache, and the server URL file -- are in its
`sc26_agent/` subdirectory:

```
/work1/proj_sc26students/shared/
└── sc26_agent/
    ├── sc26_containers/vllm-openai-rocm.sif    # pulled by pull_image.sh
    ├── sc26_model_cache/                       # populated by download_model.sh
    └── sc26_agent_server_url                   # written by start_server.sh
```

Every script in this directory, along with `setup/launch_qwen.sh`,
`setup/launch_aider.sh`, and `ask_model.py`, computes the shared directory as:

```bash
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
```

Set the `SC26_SHARED_DIR` environment variable before submitting if you need
to test against a different location (for example, a per-developer scratch
area). All scripts will pick it up.

### Permissions

`/work1` does not support ACLs, so access is controlled with two groups:
students and instructors are both in `proj_sc26students`, and only instructors
are in `sc26dev`.

| Path | Group | Mode | Students | Instructors |
|------|-------|------|----------|-------------|
| `shared/` | `proj_sc26students` | `2750` | read | read |
| `shared/sc26_agent/` | `sc26dev` | `2775` | read | read/write |
| `sc26_containers/`, `sc26_model_cache/` | `sc26dev` | `2770` | none | read/write |
| `sc26_agent_server_url` | `sc26dev` | `0664` | read | read/write |

Nobody outside `proj_sc26students` can enter `shared/`, so the "other"
permission bits on `sc26_agent/` and the URL file apply only to students. Only
the owner of `shared/` can write its top level. The scripts here use
`umask 0007` so the files they create stay instructor-only, and
`start_server.sh` publishes the URL file as `0664` so students can read it.

## URL File Lifecycle

The scripts maintain this invariant:

> **The URL file (`$SC26_SHARED_DIR/sc26_agent/sc26_agent_server_url`) exists if and only
> if a vLLM server is currently running and ready to accept requests.**

This is enforced by:

1. **`start_server.sh` refuses to launch** if the URL file already exists. This
   prevents accidentally running two GPU-burning servers at once. Use
   `scancel --name=vllm-server` to stop an existing server first.
2. **The URL is only written after vLLM is verified ready** (the script polls
   `/v1/models` until it responds).
3. **A bash `trap` removes the URL file on any exit** -- success, `scancel`,
   timeout, or error.

So if `$SC26_SHARED_DIR/sc26_agent/sc26_agent_server_url` exists, you can trust it. If it
doesn't, either no server is running or one is starting up -- check the job
log.

## Pre-Tutorial Setup (Day Before)

### Step 1: Pull the vLLM ROCm container image

This downloads the official `vllm/vllm-openai-rocm:latest` Docker image and
converts it to an Apptainer SIF file (~10 GB):

```bash
sbatch setup/vllm-server/pull_image.sh
```

The SIF file is stored at `$SC26_SHARED_DIR/sc26_agent/sc26_containers/vllm-openai-rocm.sif`.

### Step 2: Download model weights to shared storage

Pre-cache the model so the server starts quickly on tutorial day:

```bash
sbatch setup/vllm-server/download_model.sh
```

This downloads Qwen3-Coder-30B-A3B-Instruct (~60 GB) to a shared cache
directory under `$SC26_SHARED_DIR/sc26_agent/sc26_model_cache`. Takes ~15-30 minutes
depending on network speed.

### Step 3: Test the server

```bash
sbatch setup/vllm-server/start_server.sh

# Watch the startup log -- the URL file appears once vLLM is ready
tail -f vllm-server_<JOBID>.out

# Once you see "[READY] Server URL written...", from the login node:
curl $(cat "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url")/models

# Or use the included CLI helper:
python3 setup/vllm-server/ask_model.py "Write a HIP kernel that adds two vectors."
```

To stop the server: `scancel --name=vllm-server`. The trap will remove the
URL file automatically.

## Tutorial Day

### Start the server (morning, before Module 1)

The site's job-submission filter caps `mi3001x` jobs at 4 hours, so the
server runs as two chained jobs in the instructor reservation (`sc26agent`,
one `mi3001x` node, 06:30 to 17:30). At about 08:15, submit both:

```bash
JOB1=$(sbatch --parsable --reservation=sc26agent setup/vllm-server/start_server.sh)
sbatch --reservation=sc26agent --dependency=afterany:$JOB1 setup/vllm-server/start_server.sh
tail -f vllm-server_${JOB1}.out   # wait for "[READY]"
```

| Job | Runs | Covers |
|-----|------|--------|
| First | ~08:15 to 12:15 | Welcome through Module 4 |
| Second | ~12:18 to 16:18 | Module 5 through Wrap-Up |

The first job hits its time limit during lunch. Slurm sends it SIGTERM, its
cleanup trap removes the URL file, and the second job starts automatically on
the same reserved node. It is ready about 3 minutes later at the same URL, so
students only see Aider requests fail during that window. Both jobs end well
before the reservation does; Slurm kills running jobs when a reservation ends
(`ResvOverRun` is 0 on this cluster).

If the first job ends early, the second starts right away and ends 4 hours
later, before the tutorial does. Queue a third job behind it the same way,
with `--dependency=afterany:<second job ID>`.

Each job:

1. Allocates the reserved `mi3001x` node
2. Launches the vLLM ROCm container via Apptainer
3. Polls vLLM until it's ready (about 2-3 minutes with a cached model), then
   writes the URL file
4. Runs for up to 4 hours
5. On exit, automatically removes the URL file

### Tell students the URL

The server URL is auto-discovered and written to the shared file:

```bash
cat "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url"
# e.g., http://10.0.100.182:8321/v1
```

Students load it with:

```bash
export AGENT_API_URL=$(cat "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url")
```

This is already built into the Qwen Code and aider launchers and the
`ask_model.py` helper.

### Monitor the server

```bash
# Check if the job is running
squeue -u $USER --name=vllm-server

# Check server health
curl $(cat "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url")/models

# Watch the server logs
tail -f vllm-server_<JOBID>.out
```

### If the server goes down during the tutorial

Any instructor in `sc26dev` can restart the server; only step 2 needs the
job's owner or a Slurm administrator. Once the node is free, a restart takes
about 3 minutes.

1. **Check the state of the job and the URL file.**
   ```bash
   squeue --name=vllm-server        # server jobs from all users
   URL_FILE="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url"
   ls -l "$URL_FILE" && curl -sf "$(cat "$URL_FILE")/models" >/dev/null && echo healthy
   ```
2. **Job still running, but the server doesn't respond:** the job must be
   cancelled before a replacement can use the reserved node. Only its owner or
   a Slurm administrator can do this (`scancel --name=vllm-server` only cancels
   your own jobs):
   ```bash
   scancel <JOBID>
   ```
3. **Job gone, but the URL file is still there:** the node went down before
   the cleanup trap could run. Remove the stale file, or `start_server.sh` will
   refuse to start:
   ```bash
   rm "$URL_FILE"
   ```
4. **Start a replacement.** If the failed job was the first of the two
   chained jobs, the second starts automatically once it is gone; queue
   another behind it so the server lasts until the end of the day:
   ```bash
   sbatch --reservation=sc26agent --dependency=afterany:<second job ID> setup/vllm-server/start_server.sh
   ```
   Otherwise, submit one yourself. Its time limit (4 hours at most) must end
   before the reservation does (17:30):
   ```bash
   sbatch --reservation=sc26agent setup/vllm-server/start_server.sh                 # before 13:30
   sbatch --reservation=sc26agent --time=3:00:00 setup/vllm-server/start_server.sh  # e.g., at 14:00
   ```
   If the reserved node itself is down, ask a Slurm administrator to replace
   it in the reservation, or submit without `--reservation` if another
   `mi3001x` node is idle.
5. **Tell students to restart Qwen Code** (`/quit`, then run
   `launch_qwen.sh` again). The launcher reads the URL only when it starts, and
   the new server may be on a different node.

### Shut down after the tutorial

```bash
scancel --name=vllm-server
```

The URL file will be cleaned up automatically by the trap.

## Resource Impact

- Uses **1 dedicated `mi3001x` node** (1x MI300X, 192 GB HBM3)
- Qwen3-Coder-30B-A3B in BF16 uses ~60 GB; the rest (~110 GB) is KV cache,
  enabling many concurrent contexts
- vLLM's continuous batching + PagedAttention can handle 50-100+ concurrent
  requests; only ~3B parameters are active per token (MoE), so per-user
  decode latency stays low even under load
- One server instance should be sufficient for 80-100 students since agent
  requests are bursty (students aren't all querying at the exact same moment)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "URL file already exists" | A server is already running. `scancel --name=vllm-server` first. If you're sure none is running (for example, after a node crash), `rm "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url"`. See [If the server goes down during the tutorial](#if-the-server-goes-down-during-the-tutorial). |
| URL file appears but `curl` gives `Connection refused` | Shouldn't happen with the new lifecycle. If it does, the server crashed between the readiness check and accepting your request -- check the `.out` log. |
| Job is running but no URL file yet | vLLM is still loading the model. Watch `tail -f vllm-server_<JOBID>.out` -- look for `[READY]`. First load takes ~60-120s; cached loads are faster. |
| Server OOM during startup | Reduce `--max-model-len` in `start_server.sh` (currently 32768). |
| Permission denied writing to `$SC26_SHARED_DIR/sc26_agent` | Confirm with `id` that you are in both the `sc26dev` and `proj_sc26students` groups (log in again after being added), or override `SC26_SHARED_DIR` to a writable path for testing. |
| Container not found | Run `sbatch setup/vllm-server/pull_image.sh` first. |
