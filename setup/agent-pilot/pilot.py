#!/usr/bin/env python3
"""
pilot.py -- Run the Agent Practice worked examples against real agents

INSTRUCTOR-ONLY. For each task (one worked example per module) and agent, this
copies a clean tree of the repo into a scratch directory, puts the student's
finished exercise in place, runs the agent once with no person typing, and then
runs the module's own check on the result: compile, run through Slurm, and
compare with known answers. It records whether the check passed, how long the
agent took, and how many tokens it used.

Agents:
    aider        setup/launch_aider.sh (ask mode or code mode), edits only
    qwen-edit    setup/launch_qwen.sh, ask-permissions or auto-edit, no shell
    qwen-agent   like qwen-edit, but may compile and use Slurm on its own
    reference    known-good answers, no server needed: tests the checks

The agent server must be running for every agent except "reference". Each
agent gets rules.md, the rules file a student would have by Module 2 (except
in m1-norules).

Run from anywhere on the login node:
    python3 setup/agent-pilot/pilot.py --agents reference
    python3 setup/agent-pilot/pilot.py --agents aider,qwen-edit,qwen-agent --reps 5
    python3 setup/agent-pilot/pilot.py --agents qwen-agent --tasks m3-speedup \\
        --reps 50 --jobs 50 --no-check                # server load test

Developers with more than one Slurm allocation must pass --qos (or export
SBATCH_QOS). Results go to $WORK/agent-pilot/<timestamp>/: results.jsonl,
summary.txt, and every run's directory with the agent's output and diff.
"""

import argparse
import concurrent.futures as cf
import contextlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

PILOT_DIR = Path(__file__).resolve().parent
REPO = PILOT_DIR.parent.parent
REFERENCE = PILOT_DIR / "reference"
RULES = PILOT_DIR / "rules.md"

AGENTS = ["aider", "qwen-edit", "qwen-agent", "reference"]
INTERACTIVE = threading.Semaphore(3)

# Shell commands qwen-agent may run without approval: compiling and Slurm,
# never a program on the login node. --allowed-tools takes command prefixes
# as Shell(<prefix>), not the Bash(<pattern>) rules of settings files, and
# checks each part of a compound command (cd ... && ...) separately.
QWEN_AGENT_SHELL = [
    "Shell(gcc)", "Shell(mpicc)", "Shell(hipcc)",
    "Shell(srun)", "Shell(sbatch)", "Shell(squeue)", "Shell(sacct)",
    "Shell(cd)", "Shell(cat)", "Shell(ls)", "Shell(chmod)", "Shell(grep)",
    "Shell(head)", "Shell(tail)", "Shell(sleep)",
    "Shell(python3 speedup.py)", "Shell(bash sweep.sh)", "Shell(./sweep.sh)",
]


@dataclass
class Run:
    agent: str
    task: "Task"
    rep: int
    dir: Path
    tree: Path = None
    workdir: Path = None
    env: dict = None
    rules: Path = None
    output: str = ""
    notes: dict = field(default_factory=dict)


@dataclass
class Task:
    name: str
    workdir: str                      # relative to the repo tree
    mode: str                         # "ask" or "code"
    prompt: str
    files: List[str] = field(default_factory=list)   # files the agent edits
    read: List[str] = field(default_factory=list)    # read-only context (aider)
    rules: bool = True
    setup: Optional[Callable] = None
    reference: Optional[Callable] = None
    check: Optional[Callable] = None


# --- Helpers ----------------------------------------------------------------
def sh(cmd, cwd, env, timeout):
    """Run a command; return (returncode or None on timeout, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        def text(b):
            return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
        return None, text(e.stdout), text(e.stderr)


def save(run, name, out, err):
    """Keep a check's output in the run's directory for review."""
    (run.dir / f"check_{name}.txt").write_text(f"{out}\n--- stderr ---\n{err}")


def floats(text):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def copy(src, dst):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def batch(run, args, name, command, *options, timeout=1800):
    """Run a check's command in a batch job; return (exit code, job ID, output).

    The site filter allows only 4 interactive (srun) jobs per user in the
    queue, so the checks use batch jobs and leave those for the agents.
    """
    out_file = run.dir / f"check_{name}.txt"
    rc, out, err = sh(["sbatch", "--wait", "--parsable",
                       f"--partition={args.partition}", "--nodes=1",
                       f"--output={out_file}", *options, f"--wrap={command}"],
                      run.workdir, run.env, timeout)
    job = out.strip().split(";")[0] if out.strip() else None
    text = out_file.read_text(errors="replace") if out_file.exists() else err
    return rc, job, text


def job_steps(job, env):
    """How many srun steps a finished job ran (sacct can lag a few seconds)."""
    for _ in range(6):
        rc, out, _ = sh(["sacct", "-j", job, "-n", "-P", "-o", "JobID"], None, env, 30)
        ids = out.split()
        if ids:
            return sum(1 for i in ids if re.search(r"\.\d+$", i))
        time.sleep(5)
    return None


def for_partition(args, path):
    """Point a script the agent wrote at the pilot's partition."""
    if args.partition != "mi2101x":
        p = Path(path)
        p.write_text(p.read_text().replace("mi2101x", args.partition))


# --- Module 1: the same question with and without the rules file ------------
M1_PROMPT = ("Give me the srun command to run hostname on a compute node of "
             "this tutorial's partition.")


def check_m1(run, args):
    text = run.output
    notes = {"partition": "mi2101x" in text,
             "time": bool(re.search(r"--time|\s-t\s*\d", text)),
             "srun": "srun" in text}
    return all(notes.values()), notes


def reference_m1(run):
    run.output = "srun --partition=mi2101x --nodes=1 --time=1:00 hostname"


# --- Module 2: a batch script from a spec, checked by the site filter --------
def check_m2(run, args):
    script = run.workdir / "agent_job.sh"
    if not script.exists():
        return False, {"error": "agent_job.sh not written"}
    src = script.read_text()
    rc, out, err = sh(["sbatch", "--test-only", "agent_job.sh"],
                      run.workdir, run.env, 120)
    save(run, "filter", out, err)
    notes = {"filter_ok": rc == 0,
             "output_name": bool(re.search(
                 r"agent-job_(%j|\$\{?SLURM_JOB_ID\}?|\$\{?SLURM_JOBID\}?)\.out", src)),
             "nodes_1": bool(re.search(r"(--nodes|-N)[= ]?1\b", src)),
             "runs_hello": "hello_compute" in src,
             "cpus_8": bool(re.search(r"(cpus-per-task|ntasks)[= ]8\b|-c ?8\b", src)),
             "gpus_flag": bool(re.search(r"--gpus|--gres", src))}
    if rc != 0:
        reasons = re.findall(r"--> (.*)", err)
        notes["filter_error"] = " / ".join(reasons) or err.strip()[-200:]
    passed = notes["filter_ok"] and notes["output_name"] and notes["runs_hello"]
    return passed, notes


def reference_m2(run):
    copy(REFERENCE / "agent_job.sh", run.workdir / "agent_job.sh")


# --- Module 3: a speedup table from an output file -------------------------
M3_TIMES = {1: 4.012, 2: 2.031, 4: 1.024, 8: 0.533, 16: 0.291}


def setup_m3(run):
    copy(PILOT_DIR / "fixtures" / "openmp-pi_12345.txt",
         run.workdir / "openmp-pi_12345.out")


def check_m3(run, args):
    if not (run.workdir / "speedup.py").exists():
        return False, {"error": "speedup.py not written"}
    rc, out, err = sh(["python3", "speedup.py", "openmp-pi_12345.out"],
                      run.workdir, run.env, 60)
    save(run, "speedup", out, err)
    nums = floats(out)
    expected = [M3_TIMES[1] / t for t in M3_TIMES.values()]
    found = [any(abs(x - e) <= 0.051 for x in nums) for e in expected]
    notes = {"ran": rc == 0, "speedups_found": sum(found), "expected": len(expected)}
    if rc != 0:
        notes["error"] = (err.strip().splitlines() or [""])[-1][:200]
    return rc == 0 and all(found), notes


def reference_m3(run):
    copy(REFERENCE / "speedup.py", run.workdir / "speedup.py")


# --- Module 4: a ring, tested with synchronous sends ------------------------
def check_m4(run, args):
    src_path = run.workdir / "ring_agent.c"
    if not src_path.exists():
        return False, {"error": "ring_agent.c not written"}
    src = src_path.read_text()
    notes = {"sendrecv": "MPI_Sendrecv" in src,
             "nonblocking": bool(re.search(r"MPI_I(send|recv)\b", src)),
             "blocking_send": bool(re.search(r"\bMPI_Send\s*\(", src))}
    (run.workdir / "ring_ssend.c").write_text(
        re.sub(r"\bMPI_Send\s*\(", "MPI_Ssend(", src))
    rc, out, err = sh(["mpicc", "-O2", "-o", "ring_ssend", "ring_ssend.c"],
                      run.workdir, run.env, 120)
    if rc != 0:
        notes["compile_error"] = err.strip()[-300:]
        return False, notes
    passed = True
    for n in (4, 8):
        rc, job, out = batch(run, args, f"ssend_{n}", "srun ./ring_ssend",
                             f"--ntasks={n}", "--time=1:00")
        ok = rc == 0
        for r in range(n):
            m = re.search(rf"Rank\s+{r}\b[^\n]*?received\D*?(-?\d+)", out)
            ok = ok and m is not None and int(m.group(1)) == (r - 1) % n
        notes[f"ssend_{n}_ok"] = ok
        if "TIME LIMIT" in out:
            notes[f"ssend_{n}"] = "deadlock (time limit)"
        passed = passed and ok
    return passed, notes


def reference_m4(run):
    copy(REFERENCE / "ring_agent.c", run.workdir / "ring_agent.c")


# --- Module 5: an automated block-size sweep that reports failures ---------
def setup_m5(run):
    copy(run.workdir / "solutions" / "vector_scale_solution.cpp",
         run.workdir / "vector_scale.cpp")


def check_m5(run, args):
    script = run.workdir / "sweep.sh"
    if not script.exists():
        return False, {"error": "sweep.sh not written"}
    for_partition(args, script)
    # Inside the batch job each of the script's sruns becomes a job step, so
    # counting steps shows whether it really used a single srun.
    rc, job, out = batch(run, args, "sweep", "bash sweep.sh", "--time=15:00")
    steps = job_steps(job, run.env) if job else None
    rows = {}
    for line in out.splitlines():
        m = re.match(r"^\W*(\d+)\b(.*)$", line)
        if m and int(m.group(1)) in (32, 64, 128, 256, 512, 1024, 2048):
            rows[int(m.group(1))] = m.group(2)
    numbers_ok = all(
        b in rows and any(10 < x < 5000 for x in floats(rows[b]))
        for b in (32, 64, 128, 256, 512, 1024))
    row_2048 = rows.get(2048, "")
    honest = (bool(re.search(r"fail|error|invalid|n/?a|skip|not ", row_2048, re.I))
              and "PASSED" not in row_2048)
    notes = {"exit": rc, "numbers_ok": numbers_ok, "honest_2048": honest,
             "srun_steps": steps, "row_2048": row_2048.strip()[:120]}
    return numbers_ok and honest and steps == 1, notes


def reference_m5(run):
    copy(REFERENCE / "sweep.sh", run.workdir / "sweep.sh")


# --- Module 6: batched generation, checked for the padding bug --------------
def setup_m6(run):
    sol = run.workdir / "solutions" / "run_inference_solution.py"
    copy(sol, run.workdir / "run_inference.py")
    copy(sol, run.workdir / "batch_agent.py")


def check_m6(run, args):
    src = (run.workdir / "batch_agent.py").read_text()
    rc, out, err = sh(["sbatch", "--wait", f"--partition={args.partition}",
                       "submit_inference.sh", "batch_agent.py"],
                      run.workdir, run.env, 1800)
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        return False, {"error": (err.strip().splitlines() or [""])[-1][:200]}
    job = m.group(1)
    job_out = (run.workdir / f"inference_{job}.out").read_text(errors="replace")
    job_err = (run.workdir / f"inference_{job}.err").read_text(errors="replace")
    tok_s = [x for line in job_out.splitlines() if "tok/s" in line
             for x in floats(line.split("tok/s")[0][-12:])]
    notes = {"job": job, "exit": rc, "traceback": "Traceback" in job_err,
             "padding_warning": "right-padding was detected" in job_err,
             "left_padding": bool(re.search(r"padding_side\s*=\s*['\"]left", src)),
             "single_generate": len(re.findall(r"\.generate\(", src)) == 1,
             "max_tok_s": max(tok_s) if tok_s else None}
    passed = rc == 0 and not notes["traceback"] and not notes["padding_warning"]
    return passed, notes


def reference_m6(run):
    copy(run.workdir / "solutions" / "batch_inference_solution.py",
         run.workdir / "batch_agent.py")


TASKS = {t.name: t for t in [
    Task("m1-norules", "module-01-hpc-foundations", "ask", M1_PROMPT,
         rules=False, reference=reference_m1, check=check_m1),
    Task("m1-rules", "module-01-hpc-foundations", "ask", M1_PROMPT,
         reference=reference_m1, check=check_m1),
    Task("m2-batch", "module-02-slurm/exercises", "code",
         "Create agent_job.sh: a Slurm batch script that runs ./hello_compute "
         "with 8 CPU cores for at most 3 minutes and writes its output to "
         "agent-job_<job ID>.out.",
         files=["agent_job.sh"], reference=reference_m2, check=check_m2),
    Task("m3-speedup", "module-03-openmp/exercises", "code",
         "Write speedup.py, which reads an output file like openmp-pi_12345.out "
         "(given as a command-line argument) and prints a table of threads, "
         "time, and speedup (the 1-thread time divided by the N-thread time).",
         files=["speedup.py"], read=["openmp-pi_12345.out"],
         setup=setup_m3, reference=reference_m3, check=check_m3),
    Task("m4-ring", "module-04-mpi/exercises", "code",
         "Write ring_agent.c: an MPI program in which every rank sends its rank "
         "number to the next rank (the last rank sends to rank 0) and receives "
         "from the previous rank, then prints a line 'Rank <r> received <v>'. "
         "Use MPI_Send and MPI_Recv.",
         files=["ring_agent.c"], reference=reference_m4, check=check_m4),
    Task("m5-sweep", "module-05-hip/exercises", "code",
         "Write sweep.sh, which compiles vector_scale.cpp with -DBLOCK_SIZE set "
         "to 32, 64, 128, 256, 512, 1024, and 2048, runs all of them in a single "
         "srun on our partition, and prints a table of block size, bandwidth, "
         "and status.",
         files=["sweep.sh"], read=["vector_scale.cpp"],
         setup=setup_m5, reference=reference_m5, check=check_m5),
    Task("m6-batch", "module-06-ai-inference-finetuning/exercises", "code",
         "Change batch_agent.py so it generates the responses for all prompts "
         "at each temperature in a single model.generate() call, and prints "
         "the total tokens per second.",
         files=["batch_agent.py"], setup=setup_m6, reference=reference_m6,
         check=check_m6),
]}


# --- Running one agent on one task -----------------------------------------
def make_tree(run):
    """Copy the repo's tracked files (as in the working tree) into a fresh git repo."""
    run.tree = run.dir / "repo"
    files = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True,
                           capture_output=True).stdout.decode().split("\0")
    for f in filter(None, files):
        if (REPO / f).is_file():
            copy(REPO / f, run.tree / f)
    git = ["git", "-c", "user.name=pilot", "-c", "user.email=pilot@localhost"]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(git + cmd, cwd=run.tree, check=True, capture_output=True)
    run.workdir = run.tree / run.task.workdir


def agent_command(run):
    task = run.task
    if run.agent == "aider":
        # aider has no "code" chat mode flag: code mode is an edit format.
        cmd = ["bash", str(run.tree / "setup/launch_aider.sh"),
               "--chat-mode", "ask" if task.mode == "ask" else "diff",
               "--yes-always", "--no-suggest-shell-commands", "--no-pretty",
               "--analytics-disable", "--message", task.prompt]
        if task.rules:
            cmd += ["--read", str(run.rules)]
        for f in task.read:
            cmd += ["--read", f]
        return cmd + task.files
    mode = "default" if task.mode == "ask" else "auto-edit"
    cmd = ["bash", str(run.tree / "setup/launch_qwen.sh"),
           "--approval-mode", mode, "--output-format", "json",
           "--max-wall-time", "15m", "-p", task.prompt]
    if run.agent == "qwen-agent" and task.mode == "code":
        cmd += ["--allowed-tools", ",".join(QWEN_AGENT_SHELL)]
    return cmd


def qwen_text(stdout):
    """The final answer from Qwen Code's JSON output (after the launcher's banner)."""
    m = re.search(r"^[\[{]", stdout, re.M)
    try:
        data = json.loads(stdout[m.start():]) if m else None
    except ValueError:
        data = None
    if data is None:
        return ""
    items = data if isinstance(data, list) else [data]
    for item in reversed(items):
        if isinstance(item, dict):
            for key in ("result", "response"):
                if isinstance(item.get(key), str):
                    return item[key]
    return stdout


def usage(run, stdout):
    """Tokens and requests: from aider's Tokens lines or Qwen Code's usage record."""
    u = {"tokens_in": 0, "tokens_out": 0, "tokens_cached": None, "requests": None,
         "tool_calls": None}
    if run.agent == "aider":
        for sent, k1, recv, k2 in re.findall(
                r"Tokens: ([\d.]+)(k?) sent, ([\d.]+)(k?) received", stdout):
            u["tokens_in"] += float(sent) * (1000 if k1 else 1)
            u["tokens_out"] += float(recv) * (1000 if k2 else 1)
        u["requests"] = stdout.count("Tokens: ")
        return u
    record = Path(run.env["QWEN_HOME"]) / "usage_record.jsonl"
    if record.exists():
        u.update(tokens_cached=0, requests=0, tool_calls=0)
        for line in record.read_text().splitlines():
            rec = json.loads(line)
            for m in rec.get("models", {}).values():
                u["tokens_in"] += m.get("inputTokens", 0)
                u["tokens_out"] += m.get("outputTokens", 0)
                u["tokens_cached"] += m.get("cachedTokens", 0)
                u["requests"] += m.get("requests", 0)
            u["tool_calls"] += rec.get("tools", {}).get("totalCalls", 0)
    return u


def do_run(run, args):
    result = {"agent": run.agent, "task": run.task.name, "rep": run.rep,
              "dir": str(run.dir)}
    try:
        make_tree(run)
        run.env = dict(os.environ, QWEN_HOME=str(run.dir / "qwen_home"))
        if args.qos:
            run.env["SBATCH_QOS"] = run.env["SLURM_QOS"] = args.qos
        Path(run.env["QWEN_HOME"]).mkdir(parents=True)
        if run.task.rules:
            # Module 1 asks about the real partition; the rest run on --partition.
            rules = RULES.read_text()
            if not run.task.name.startswith("m1"):
                rules = rules.replace("mi2101x", args.partition)
            run.rules = run.dir / "rules.md"
            run.rules.write_text(rules)
            copy(run.rules, Path(run.env["QWEN_HOME"]) / "QWEN.md")
        if run.task.setup:
            run.task.setup(run)

        t0 = time.time()
        if run.agent == "reference":
            run.task.reference(run)
            rc = 0
        else:
            # qwen-agent may srun on its own; stay under the site's limit of
            # 4 interactive jobs per user (one is the server's allocation).
            limit = (INTERACTIVE if run.agent == "qwen-agent" and run.task.mode == "code"
                     else contextlib.nullcontext())
            with limit:
                t0 = time.time()
                rc, out, err = sh(agent_command(run), run.workdir, run.env,
                                  args.agent_timeout)
            (run.dir / "agent.out").write_text(out)
            (run.dir / "agent.err").write_text(err)
            run.output = qwen_text(out) if run.agent.startswith("qwen") else out
            result.update(usage(run, out))
        result["agent_time"] = round(time.time() - t0, 1)
        result["agent_rc"] = rc

        subprocess.run(["git", "add", "-A", "-N"], cwd=run.tree, capture_output=True)
        diff = subprocess.run(["git", "diff"], cwd=run.tree, capture_output=True,
                              text=True).stdout
        (run.dir / "diff.txt").write_text(diff)

        if args.no_check:
            result["passed"] = None
        else:
            t0 = time.time()
            passed, notes = run.task.check(run, args)
            result.update(passed=passed, notes=notes,
                          check_time=round(time.time() - t0, 1))
    except Exception as e:  # keep going; one broken run shouldn't stop the pilot
        result.update(passed=False, error=f"{type(e).__name__}: {e}")
    return result


# --- Summary -----------------------------------------------------------------
def summarize(results):
    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    def fmt(x, spec):
        return format(x, spec) if x is not None else "-".rjust(int(spec.split(".")[0]))

    lines = [f"{'agent':<11} {'task':<11} {'pass':>6} {'time med':>8} {'max':>6} "
             f"{'tok in':>7} {'tok out':>7} {'cached':>6} {'reqs':>5} {'tools':>5}"]
    keys = sorted({(r["agent"], r["task"]) for r in results},
                  key=lambda k: (AGENTS.index(k[0]), list(TASKS).index(k[1])))
    for agent, task in keys:
        rs = [r for r in results if r["agent"] == agent and r["task"] == task]
        checked = [r for r in rs if r.get("passed") is not None]
        passed = sum(1 for r in checked if r["passed"])
        times = [r.get("agent_time") for r in rs]
        tmax = max([t for t in times if t is not None], default=None)
        tin = med([r.get("tokens_in") for r in rs])
        cached = med([r["tokens_cached"] / r["tokens_in"] for r in rs
                      if r.get("tokens_cached") is not None and r.get("tokens_in")])
        lines.append(
            f"{agent:<11} {task:<11} {f'{passed}/{len(checked)}':>6} "
            f"{fmt(med(times), '8.1f')} {fmt(tmax, '6.0f')} "
            f"{fmt(tin, '7.0f')} {fmt(med([r.get('tokens_out') for r in rs]), '7.0f')} "
            f"{fmt(cached * 100 if cached is not None else None, '5.0f')}"
            f"{'%' if cached is not None else ' '} "
            f"{fmt(med([r.get('requests') for r in rs]), '5.0f')} "
            f"{fmt(med([r.get('tool_calls') for r in rs]), '5.0f')}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", default="aider,qwen-edit,qwen-agent",
                   help=f"comma-separated, from: {', '.join(AGENTS)}")
    p.add_argument("--tasks", default="all",
                   help=f"comma-separated, from: {', '.join(TASKS)} (default: all)")
    p.add_argument("--reps", type=int, default=1, help="runs per agent and task")
    p.add_argument("--jobs", type=int, default=4, help="runs at the same time")
    p.add_argument("--partition", default="mi2101x",
                   help="partition for the checks' Slurm jobs (e.g. devel)")
    p.add_argument("--qos", default=os.environ.get("SBATCH_QOS"),
                   help="Slurm QOS for the agents' and checks' jobs")
    p.add_argument("--agent-timeout", type=int, default=1200, help="seconds")
    p.add_argument("--no-check", action="store_true",
                   help="skip the checks (for load tests)")
    args = p.parse_args()

    agents = args.agents.split(",")
    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    for a in agents:
        if a not in AGENTS:
            sys.exit(f"Unknown agent: {a}")
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"Unknown task: {t}")
    if not os.environ.get("WORK"):
        sys.exit("ERROR: The WORK environment variable is not set.")

    out_dir = (Path(os.environ["WORK"]) / "agent-pilot"
               / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    runs = [Run(a, TASKS[t], k, out_dir / a / t / f"rep{k:02d}")
            for k in range(1, args.reps + 1) for t in tasks for a in agents]
    print(f"{len(runs)} runs, {args.jobs} at a time. Output: {out_dir}", flush=True)

    results = []
    with open_results(out_dir) as log, cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(do_run, r, args) for r in runs]
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
            status = {True: "PASS", False: "FAIL", None: "done"}[r.get("passed")]
            print(f"[{len(results)}/{len(runs)}] {status} {r['agent']} {r['task']} "
                  f"rep{r['rep']:02d} ({r.get('agent_time', '-')} s) "
                  f"{r.get('notes', r.get('error', ''))}", flush=True)

    summary = summarize(results)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)


def open_results(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    return open(out_dir / "results.jsonl", "w")


if __name__ == "__main__":
    main()
