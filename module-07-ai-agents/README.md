# Module 7 -- AI Agents & Capstone

<div class="event-meta module-meta">
  <span><strong>3:00 to 3:50 PM CST</strong></span>
  <span>50 min total</span>
  <span>~15 min lecture · ~35 min hands-on</span>
</div>

## Learning Objectives

By the end of this module, you will be able to:

- Explain what an AI agent is and how it differs from a simple chatbot
- Describe the agent loop: perceive, reason, act, observe
- Build a simple agent that uses an LLM to generate and execute code
- Reflect on how AI agents helped (and occasionally misled) you today

---

## Key Concepts

### What is an AI Agent?

You've been using AI assistants all day to help write code. But what makes
an AI tool an **agent** rather than just a chatbot?

| Chatbot | Agent |
|---------|-------|
| Takes a question, gives an answer | Takes a **goal**, works toward it over multiple steps |
| No access to tools | Can **call tools**: run code, read files, search the web |
| Single turn | **Multi-turn loop**: reason → act → observe → reason again |
| You do the work | It does the work (with your oversight) |

An agent is: **LLM + Tools + A Loop**

### The Agent Loop

```
         ┌─────────────────────────────────────┐
         │                                     │
         ▼                                     │
    ┌─────────┐     ┌─────────┐     ┌──────────┴──┐
    │ Observe │────►│  Think  │────►│     Act      │
    │ (read   │     │ (LLM    │     │ (call tool:  │
    │ result) │     │ reasons)│     │  run code,   │
    └─────────┘     └─────────┘     │  read file)  │
         ▲                          └──────────────┘
         │                                │
         └────────────────────────────────┘
              Loop until goal is reached
```

1. **Observe**: The agent reads the current state (tool outputs, errors, files)
2. **Think**: The LLM reasons about what to do next
3. **Act**: The agent calls a tool (e.g., runs a shell command)
4. **Repeat**: The tool's output becomes the next observation

### The Connection to Inference

Every "Think" step is an **LLM inference call** -- exactly what you learned in
Module 6. An agent might make 5-20 inference calls to complete a single task.
This is why fast inference matters: a slow model means a slow agent.

### Tool Use

Agents interact with the world through **tools** -- functions the LLM can call.
Common tools include:

| Tool | What it does |
|------|-------------|
| `run_command(cmd)` | Execute a shell command and return output |
| `read_file(path)` | Read a file's contents |
| `write_file(path, content)` | Write content to a file |
| `search(query)` | Search the web or a codebase |

The LLM decides **which** tool to call and **what arguments** to pass, based on
the conversation so far.

### How Agents Work Under the Hood

When you use an AI coding assistant (Cursor, Copilot, etc.), here's what happens:

1. Your request + context is sent to an LLM
2. The LLM responds with either text **or a tool call** (structured JSON)
3. If it's a tool call, the system executes it and sends the result back
4. The LLM sees the result and decides the next step
5. This continues until the task is done

```json
{
  "tool": "run_command",
  "arguments": {
    "command": "gcc -fopenmp -O2 -o pi_openmp pi_openmp.c -lm"
  }
}
```

### Responsible Use

Agents are powerful but imperfect:

- **Hallucinations**: The LLM may generate plausible-sounding but wrong code
- **Verification**: Always check that generated code does what you expect
- **Security**: Don't let agents run arbitrary commands on production systems
- **Understanding**: An agent can write code for you, but if you don't
  understand it, you can't debug it when things go wrong

This is why we taught you the HPC fundamentals **first** today -- so you can
be a better judge of what an AI agent produces.

---

## Hands-On Exercises (~35 min)

First, navigate to the exercises directory for this module:

```bash
cd module-07-ai-agents/exercises
```

````{note}
These exercises use a shared inference server (vLLM) that your instructors have
deployed on a dedicated MI300X compute node. It serves
Qwen3-Coder-30B-A3B-Instruct via an OpenAI-compatible API.

The batch scripts auto-discover the server URL from a shared file. To check it
manually:

```bash
cat "$WORK/sc26_agent_server_url"
```

If you want to test interactively, set the environment variable:

```bash
export AGENT_API_URL=$(cat "$WORK/sc26_agent_server_url")
```
````

These scripts use the same Python venv you created in Getting Started Step 4.
If `"$WORK/sc26_venv"` is missing, go back to that setup step before submitting
the agent jobs.

### Step 0: Look at the Example

Examine the minimal agent loop:

```bash
cat ../examples/simple_agent.py
```

This shows the complete pattern: prompt the LLM, parse tool calls, execute
them, feed results back. Run it to see it in action:

```bash
sbatch submit_agent.sh
```

---

### Exercise 1: Build Your Own Agent (Part A -- Core)

Open the exercise template:

```bash
cat build_agent.py
```

There are **3 TODOs**:

1. **TODO 1**: Implement the `run_command` tool (execute a shell command safely)
2. **TODO 2**: Send a request to the LLM API with the conversation history
3. **TODO 3**: Parse the LLM's response -- detect tool calls and extract arguments

After filling in the TODOs, test your agent:

```bash
sbatch submit_agent.sh
```

Try asking your agent to:
- *"Write a C program that prints 'Hello from HPC!' and compile it"*
- *"Check what GPU is on this node using rocm-smi"*
- *"Write an OpenMP program that computes pi, compile it, and run it with 4 threads"*

**Questions:**
- Does the agent recover from errors (e.g., compilation failures)?
- Does it produce correct code on the first try?
- How many LLM calls does it take to complete a task?

---

### Exercise 2: Use a Real CLI Agent (Core)

The agent you just built is a minimal example. The CLI agent you've been using
all day -- **aider** -- is essentially the same idea, but with many more tools
(read/write files, apply diffs, run tests, integrate with git) and much more
careful prompt engineering and error recovery.

In other words: you've already used the production version of what you just
built. Now let's open the hood.

Compare what aider does against your `build_agent.py`:

| Feature | Your agent | aider |
|---|---|---|
| Backend LLM | Same Qwen3-Coder via vLLM | Same Qwen3-Coder via vLLM |
| Tool: run shell command | yes | yes |
| Tool: read file | no | yes |
| Tool: edit file (apply diff) | no | yes |
| Tool: git integration | no | yes (auto-commits, undo) |
| Streaming responses | no | yes |
| Multi-file context | no | yes |

Try driving aider through a small task. From the repo root:

```bash
cd module-03-openmp/exercises
bash ../../setup/launch_aider.sh

# Inside aider, type:
> Add OpenMP directives to pi_serial.c to parallelize the main loop using a reduction. Save it as pi_openmp_aider.c.
```

Watch what happens: aider reads the file, proposes a diff, asks you to confirm,
and writes the new file. That's the same observe-think-act loop your
`build_agent.py` ran -- it just has nicer tools.

---

### Exercise 3: Capstone Challenge (Part B -- Extension)

Now put the whole day together. Use **aider** (or your own agent if you'd
like a real challenge) to solve this multi-step problem:

> **Challenge:** Write a HIP kernel that computes the dot product of two vectors.
> The kernel should use parallel reduction within a block. Compile it, run it on
> the GPU via Slurm, and verify the result against a CPU reference.

Recommended workflow with aider:

```bash
mkdir -p ~/capstone && cd ~/capstone
bash <repo-path>/setup/launch_aider.sh
```

Then ask aider to:

1. Generate the HIP code (`> Write dot_product.cpp that ...`).
2. Review the code -- does the reduction look correct? Are there race conditions?
3. Compile with `hipcc` (`> Compile dot_product.cpp with hipcc -O2`).
4. Write a Slurm batch script (`> Write submit_dot.sh that runs ./dot_product on mi2101x`).
5. Submit, check output, iterate.

See `capstone.md` for the full description.

```{tip}
As you go, ask yourself the same questions you've been asking all day -- did
the agent get the GPU memory management right? Did it use the right reduction
pattern? Did the launch configuration make sense? You verify; the agent
assists.
```

---

### Reflection

Take a few minutes to reflect on how AI agents helped you today:

1. When did the agent give you **correct, useful** help?
2. When did it **mislead** you or produce something wrong?
3. Did you catch the mistakes? How?
4. Would you have been able to verify the agent's output if you **didn't**
   understand the fundamentals (OpenMP, MPI, HIP)?

> The goal isn't to avoid using AI agents -- they're incredibly useful tools.
> The goal is to **use them effectively**, which means understanding enough to
> know when they're right and when they're not.

---

## Quick Reference

| Concept | Description |
|---------|-------------|
| **Agent** | LLM + tools + a reasoning loop |
| **Tool** | A function the LLM can call (run code, read files, etc.) |
| **Agent loop** | Observe → Think → Act → Repeat |
| **System prompt** | Instructions that tell the LLM how to behave as an agent |
| **Tool call** | Structured output from the LLM requesting a tool execution |
| **Hallucination** | When the LLM generates confident but incorrect output |

---

**That's a wrap!** Head back to the main session for the [Wrap-Up](../README.md).
