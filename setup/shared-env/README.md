# Shared Student Environments -- Instructor Guide

The scripts in this directory build, once, the software that every student
uses, in the tutorial's shared directory. Students don't install anything:
their jobs activate the shared environments directly.

## What Gets Built

```
/work1/proj_sc26students/shared/                  ($SC26_SHARED_DIR)
├── sc26_venv -> sc26_venv-<timestamp>/           ML stack for Modules 6-7 (build_venv.sh)
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

## Rebuilding and Rolling Back

Each build goes into a new timestamped directory, and the symlink is switched
to it in one step only after its checks pass, so rebuilding never breaks
jobs that are using the current build. To roll back, point the symlink at an
older build:

```bash
cd /work1/proj_sc26students/shared
ln -sfn sc26_venv-<older-timestamp> sc26_venv
```

Older builds are listed at the end of each build log. Remove them when you no
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
```
