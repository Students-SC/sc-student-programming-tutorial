# Shared Student Environments -- Instructor Guide

The scripts in this directory build, once, the software that every student
uses, in the tutorial's shared directory. Students don't install anything:
their jobs activate the shared environments directly.

## What Gets Built

```
/work1/proj_sc26students/shared/                  ($SC26_SHARED_DIR)
├── sc26_venv -> sc26_venv-<timestamp>/           ML stack for Modules 6-7 (build_venv.sh)
├── sc26_aider_venv -> sc26_aider_venv-<timestamp>/   aider CLI agent (build_aider_venv.sh)
├── sc26_qwen_code -> sc26_qwen_code-<timestamp>/     Qwen Code CLI agent (build_qwen_code.sh)
├── sc26_hf_cache/                                Module 6 models and dataset (download_hf_cache.sh)
│   ├── hub/                                      TinyLlama, DistilBERT, and IMDB's raw files
│   └── datasets/                                 IMDB, prepared
└── sc26_agent/                                   agent server files (see ../vllm-server/README.md)
```

Every script computes the shared directory as
`${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}`, so you can build into
a scratch location for testing by exporting `SC26_SHARED_DIR` first.

## Who Can Build

Only the owner of `$SC26_SHARED_DIR` can write its top level, so only the
owner can run these scripts; each one checks this first. The scripts use
`umask 0027`, so everything they create is readable by the
`proj_sc26students` group (students and instructors) but writable only by
the owner, and each build checks those permissions before it finishes. See
[Permissions](../vllm-server/README.md#permissions) for the full model.

## Build the ML venv

From the repo root:

```bash
sbatch setup/shared-env/build_venv.sh
sbatch -p devel setup/shared-env/build_venv.sh    # if mi2101x is busy
```

The job installs exactly the versions in `requirements-sc26-venv.txt`
(PyTorch for ROCm 7.2, transformers, datasets, peft, and so on), precompiles
every module (students can't write `__pycache__`), checks that PyTorch sees
the GPU and that permissions are right, and then points `sc26_venv` at the
new build. It takes about 6 minutes and produces a 16 GB venv.

To change a package version, edit `requirements-sc26-venv.txt` and rebuild.
Don't `pip install` into the shared venv.

## Build the aider venv

aider, the CLI coding agent, has its own venv, separate from the ML venv,
because it pins older `openai` and `pydantic` versions than the
PyTorch/transformers stack. It is a pure `pip` install, so run it on the
login node:

```bash
bash setup/shared-env/build_aider_venv.sh
```

It installs exactly the versions in `requirements-sc26-aider-venv.txt`
(aider 0.86.2 and all of its dependencies), takes about a minute, and
produces a 654 MB venv. To upgrade aider, see the comment at the top of that
file.

## Install Qwen Code

Qwen Code, a CLI coding agent that reads files, runs commands, and edits code
on its own, ships as a standalone archive with its own Node.js runtime. Run
this on the login node:

```bash
bash setup/shared-env/build_qwen_code.sh
```

It downloads the release pinned in the script (`QWEN_CODE_VERSION`), checks
its SHA-256 checksum, unpacks it, and copies in `qwen-code-defaults.json`,
which `setup/launch_qwen.sh` gives Qwen Code as its lowest-precedence
settings: ask-permissions mode by default, only the plan, ask-permissions, and
auto-edit modes offered, the server's 32K context window, and no
auto-update, usage statistics, or telemetry. It takes about 20 seconds and
produces a 324 MB install. Re-run it after changing the version or the
defaults file.

The defaults also turn off features that cost context-window space and server
time without helping the tutorial. Measured in a student's clone on the
interactive screen, a one-word question took 16,274 prompt tokens per request,
plus 4 background requests (28,291 tokens) from auto-memory. With these
features off it takes 11,536 tokens and no background requests:

- **Auto-memory** (`memory.enableManagedAutoMemory`, `enableManagedAutoDream`):
  background requests after every turn, and memory files under `~/.qwen`.
  Students keep their notes in the repository's `QWEN.md` instead.
- **Bundled skills** (`skills.disabledLevels`): a catalog of about 3,000
  characters added to every conversation.
- **Unused tools** (`tools.disabled`): `skill`, `artifact`, `notebook_edit`,
  `display_image`, and `enter_plan_mode` (students switch modes with
  Shift+Tab). Don't turn off `tool_search`: without it Qwen Code declares
  every deferred tool up front, and requests grow to about 21,800 tokens.

Qwen Code calls tools natively, so the vLLM server must run with tool
calling enabled, as `start_server.sh` does.

## Download Module 6's Models and Dataset

This script uses the shared ML venv, so build that first:

```bash
sbatch setup/shared-env/download_hf_cache.sh
sbatch -p devel setup/shared-env/download_hf_cache.sh    # if mi2101x is busy
```

It downloads TinyLlama-1.1B-Chat and DistilBERT into `sc26_hf_cache/hub/`,
downloads and prepares the IMDB dataset into `sc26_hf_cache/datasets/`, and
checks that everything loads offline. It takes about a minute (2.4 GB of
models and raw files, plus 128 MB for the prepared dataset). Keep its model
names in sync with `MODEL_NAME` in the Module 6 scripts.

Students' Module 6 jobs load everything offline (`HF_HUB_OFFLINE=1`), with
the models and the dataset handled differently:

- **Models** load straight from the shared, read-only `hub/` cache
  (`HF_HUB_CACHE`).
- **The dataset** can't load from a read-only cache: the `datasets` library
  writes lock and cache files next to the data, and offline it won't rebuild
  the dataset from the raw files in `hub/`. So each student's job copies the
  prepared `datasets/` cache (128 MB) into `$WORK` once and points
  `HF_DATASETS_CACHE` at the copy. Student code still calls
  `load_dataset("imdb")`.

## Rebuilding and Rolling Back

Each build goes into a new timestamped directory, and the symlink is switched
to it in one step only after its checks pass, so rebuilding never breaks
jobs that are using the current build. To roll back, point the symlink at an
older build:

```bash
cd /work1/proj_sc26students/shared
ln -sfn sc26_venv-<older-timestamp> sc26_venv
```

The aider venv and Qwen Code work the same way
(`sc26_aider_venv-<timestamp>`, `sc26_qwen_code-<timestamp>`). Older
builds are listed at the end of each build log. Remove them when you no
longer need them.

## Check It as a Student

Owner permission bits bypass group checks, so check access as
`scstudenttest`, never as the owner:

```bash
S=/work1/proj_sc26students/shared
source "$S/sc26_venv/bin/activate"
python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
touch "$S/sc26_venv/x"          # Permission denied
deactivate
"$S/sc26_aider_venv/bin/aider" --version
"$S/sc26_qwen_code/bin/qwen" --version
ls "$S/sc26_hf_cache/hub" "$S/sc26_hf_cache/datasets"
```
