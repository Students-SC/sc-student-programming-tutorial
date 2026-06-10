# vLLM Inference Server -- Instructor Deployment Guide

This directory contains scripts to deploy a shared vLLM inference server for
Module 7 (AI Agents). The server runs inside the official `vllm/vllm-openai-rocm`
Docker image (via Apptainer/Singularity) on a dedicated MI300X compute node,
providing an OpenAI-compatible API that all students can query concurrently.

## Architecture

```
  ┌───────────────┐         ┌───────────────────────────────────┐
  │  Login Node   │         │  Dedicated mi3001x Compute Node   │
  │               │         │                                   │
  │  Students     │  HTTP   │  Apptainer Container              │
  │  submit jobs  │────────►│  vLLM + ROCm + PyTorch            │
  │               │         │  Qwen3-Coder-30B-A3B-Instruct     │
  │               │         │  Port 8321                        │
  └───────────────┘         │  1x AMD Instinct MI300X (192 GB)  │
         ▲                  └───────────────────────────────────┘
         │                              ▲
         │ HTTP                         │
  ┌──────┴────────┐                     │
  │ Student       │  HTTP               │
  │ Compute Nodes │─────────────────────┘
  │ (agent jobs)  │
  └───────────────┘
```

All nodes are on the same internal network, so HTTP requests between them
work directly. The model is a 30B-parameter Mixture-of-Experts coder model
with only ~3B parameters active per token, so inference is fast even under
heavy concurrent load.

## Shared Tutorial Directory

All resources that need to be visible to multiple people -- the container SIF,
the model cache, and the server URL file -- live under a single shared
directory. The default is:

```
/work1/sc26dev/shared/
├── sc26_containers/vllm-openai-rocm.sif    # pulled by pull_image.sh
├── sc26_model_cache/                       # populated by download_model.sh
└── sc26_agent_server_url                   # written by start_server.sh
```

Every script in this directory (and `setup/launch_aider.sh`,
`module-07-ai-agents/exercises/submit_agent.sh`) computes this path as:

```bash
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/sc26dev/shared}"
```

Set the `SC26_SHARED_DIR` environment variable before submitting if you need
to test against a different location (for example, a per-developer scratch
area). All scripts will pick it up.

## URL File Lifecycle

The scripts maintain this invariant:

> **The URL file (`$SC26_SHARED_DIR/sc26_agent_server_url`) exists if and only
> if a vLLM server is currently running and ready to accept requests.**

This is enforced by:

1. **`start_server.sh` refuses to launch** if the URL file already exists. This
   prevents accidentally running two GPU-burning servers at once. Use
   `scancel --name=vllm-server` to stop an existing server first.
2. **The URL is only written after vLLM is verified ready** (the script polls
   `/v1/models` until it responds).
3. **A bash `trap` removes the URL file on any exit** -- success, `scancel`,
   timeout, or error.

So if `$SC26_SHARED_DIR/sc26_agent_server_url` exists, you can trust it. If it
doesn't, either no server is running or one is starting up -- check the job
log.

## Pre-Tutorial Setup (Day Before)

### Step 1: Pull the vLLM ROCm container image

This downloads the official `vllm/vllm-openai-rocm:latest` Docker image and
converts it to an Apptainer SIF file (~10 GB):

```bash
sbatch setup/vllm-server/pull_image.sh
```

The SIF file is stored at `$SC26_SHARED_DIR/sc26_containers/vllm-openai-rocm.sif`.

### Step 2: Download model weights to shared storage

Pre-cache the model so the server starts quickly on tutorial day:

```bash
sbatch setup/vllm-server/download_model.sh
```

This downloads Qwen3-Coder-30B-A3B-Instruct (~60 GB) to a shared cache
directory under `$SC26_SHARED_DIR/sc26_model_cache`. Takes ~15-30 minutes
depending on network speed.

### Step 3: Test the server

```bash
sbatch setup/vllm-server/start_server.sh

# Watch the startup log -- the URL file appears once vLLM is ready
tail -f vllm-server_<JOBID>.out

# Once you see "[READY] Server URL written...", from the login node:
curl $(cat "$SC26_SHARED_DIR/sc26_agent_server_url")/models

# Or use the included CLI helper:
python3 setup/vllm-server/ask_model.py "Write a HIP kernel that adds two vectors."
```

To stop the server: `scancel --name=vllm-server`. The trap will remove the
URL file automatically.

## Tutorial Day

### Start the server (morning, before Module 1)

```bash
sbatch setup/vllm-server/start_server.sh
tail -f vllm-server_<JOBID>.out   # wait for "[READY]"
```

The job:

1. Allocates a dedicated `mi3001x` node
2. Launches the vLLM ROCm container via Apptainer
3. Polls vLLM until it's ready, then writes the URL file
4. Runs for up to 8 hours (covers the full tutorial day)
5. On exit, automatically removes the URL file

### Tell students the URL

The server URL is auto-discovered and written to the shared file:

```bash
cat "${SC26_SHARED_DIR:-/work1/sc26dev/shared}/sc26_agent_server_url"
# e.g., http://10.0.100.182:8321/v1
```

Students load it with:

```bash
export AGENT_API_URL=$(cat "${SC26_SHARED_DIR:-/work1/sc26dev/shared}/sc26_agent_server_url")
```

This is already built into the Module 7 submit scripts and the `ask_model.py`
helper.

### Monitor the server

```bash
# Check if the job is running
squeue -u $USER --name=vllm-server

# Check server health
curl $(cat "${SC26_SHARED_DIR:-/work1/sc26dev/shared}/sc26_agent_server_url")/models

# Watch the server logs
tail -f vllm-server_<JOBID>.out
```

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
| "URL file already exists" | A server is already running. `scancel --name=vllm-server` first. If you're sure none is running, `rm "$SC26_SHARED_DIR/sc26_agent_server_url"`. |
| URL file appears but `curl` gives `Connection refused` | Shouldn't happen with the new lifecycle. If it does, the server crashed between the readiness check and accepting your request -- check the `.out` log. |
| Job is running but no URL file yet | vLLM is still loading the model. Watch `tail -f vllm-server_<JOBID>.out` -- look for `[READY]`. First load takes ~60-120s; cached loads are faster. |
| Server OOM during startup | Reduce `--max-model-len` in `start_server.sh` (currently 32768). |
| Permission denied writing to `$SC26_SHARED_DIR` | Confirm with cluster admins that you have write access to the shared tutorial directory (default `/work1/sc26dev/shared`), or override `SC26_SHARED_DIR` to a writable path for testing. |
| Container not found | Run `sbatch setup/vllm-server/pull_image.sh` first. |
