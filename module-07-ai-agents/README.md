# Module 7 -- AI Agents & Capstone

> **Time:** 3:00 to 3:50 PM CST · 50 min total · ~15 min lecture · ~35 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain what an AI agent is and how it differs from a simple chatbot
- Describe the agent loop: perceive, reason, act, observe
- Build a simple agent that uses an LLM to generate and execute code
- Direct a production coding agent: give it context, approve its actions, and
  hold it to a written spec
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

When you use an AI coding agent (Qwen Code, Cursor, Copilot, etc.), here's what
happens:

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
- **Security**: Approve commands deliberately; don't let agents run arbitrary
  commands on shared or production systems
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

The batch scripts auto-discover the server URL from a shared file under the
tutorial project's shared directory (default `/work1/proj_sc26students/shared/`).
To check it manually:

```bash
cat "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url"
```

If you want to test interactively, set the environment variable:

```bash
export AGENT_API_URL=$(cat "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_agent/sc26_agent_server_url")
```
````

These scripts use the tutorial's shared Python environment, the same one as
Module 6. If a script reports that it is missing, ask an instructor.

### Exercise 1: Run a Simple Agent

#### Step 1: Examine the Agent Loop

Examine the minimal agent loop:

```bash
cat ../examples/simple_agent.py
```

This shows the complete pattern: prompt the LLM, parse tool calls, execute
them, and feed results back.

#### Step 2: Run the Agent

Run it to see the loop in action:

```bash
sbatch submit_agent.sh
```

---

### Exercise 2: Build Your Own Agent (Part A)

#### Step 1: Open the Template

Open the exercise template:

```bash
cat build_agent.py
```

#### Step 2: Complete the TODOs

There are **3 TODOs**:

1. **TODO 1**: Implement the `run_command` tool (execute a shell command safely)
2. **TODO 2**: Send a request to the LLM API with the conversation history
3. **TODO 3**: Parse the LLM's response -- detect tool calls and extract arguments

#### Step 3: Test Your Agent

After filling in the TODOs, test your agent:

```bash
sbatch submit_agent.sh
```

#### Step 4: Try Several Tasks

Try asking your agent to:
- *"Write a C program that prints 'Hello from HPC!' and compile it"*
- *"Check what GPU is on this node using rocm-smi"*
- *"Write an OpenMP program that computes pi, compile it, and run it with 4 threads"*

**Questions:**
- Does the agent recover from errors (e.g., compilation failures)?
- Does it produce correct code on the first try?
- How many LLM calls does it take to complete a task?

---

### Exercise 3: Open the Hood of a Real CLI Agent

The agent you just built is a minimal example. **Qwen Code**, the agent you
have used all day, is the same idea -- an LLM, tools, and a loop -- with many
more tools, careful prompts, and you approving its actions.

#### Step 1: Compare the Agents

| Feature | Your agent | Qwen Code |
|---|---|---|
| Backend LLM | Same Qwen3-Coder via vLLM | Same Qwen3-Coder via vLLM |
| How the model asks for a tool | JSON in a code block, which your code parses | The model's built-in tool-call format, which vLLM parses |
| Tool: run a shell command | yes | yes, after you approve it |
| Tools: read, search, and edit files | no | yes |
| Context it always starts with | your system prompt | the repository's `QWEN.md`, including the notes you added |
| Approval before acting | no | plan, ask-permissions, and auto-edit modes |
| Undo | no | `/restore` |

#### Step 2: Watch the Loop

Start Qwen Code in the Module 3 exercises directory:

```bash
cd ../../module-03-openmp/exercises
bash ../../setup/launch_qwen.sh
```

Type `/tools` to see every tool the model can call. Then ask it:

> Parallelize the main loop of pi_serial.c with OpenMP using a reduction, save it as pi_openmp_agent.c, compile it, and test it on a compute node with 1 and 16 threads.

Each action it asks you to approve is one turn of the loop you built in
Exercise 2: the model picks a tool and its arguments, Qwen Code runs it, and
the result goes back to the model. Approve only what you would run yourself;
the test belongs on a compute node, not the login node.

#### Step 3: Count the Calls and Check the Code

Type `/stats model` to see how many requests and tokens the task took.

**Questions:**
- How does that compare with the number of LLM calls your own agent needed in
  Exercise 2?
- Compare `pi_openmp_agent.c` with your `pi_openmp.c` from Module 3. Did the
  agent use the same pragmas? Did it record the thread count? Is anything
  shared that should be private?

---

### Exercise 4: Capstone Challenge

Now put the whole day together the way you would on the job: write down what
"done" means before the agent writes any code, then hold the agent to it.

> **Challenge:** Write a HIP kernel that computes the dot product of two vectors.
> The kernel should use parallel reduction within a block. Compile it, run it on
> the GPU via Slurm, and verify the result against a CPU reference.

#### Step 1: Write Your Acceptance Checks

Go back to this module's exercises directory, which is inside the repository
so the agent reads `QWEN.md`, and write down how you will know the program is
right:

```bash
cd ~/sc-student-programming-tutorial/module-07-ai-agents/exercises
cat > spec.md << 'EOF'
# Dot product: done means
- Matches a CPU reference, computed in double precision, to a relative error below 1e-4
- Works when N is not a multiple of the block size (try N = 1000003)
- Uses more than one block and combines the blocks' partial sums
- Runs on a compute node through Slurm, never on the login node
EOF
```

Add at least one check of your own, for example from the review list in
`capstone.md` (in this directory).

#### Step 2: Hand It to the Agent

```bash
bash ../../setup/launch_qwen.sh
```

Ask it to read `spec.md`, then write, compile, and test `dot_product.cpp`
against it. Once you trust its edits, you can press Shift+Tab to switch to
auto-edit mode; it still asks before every command.

#### Step 3: Hold It to Your Spec

Before you accept the result, check every line of `spec.md` yourself: read the
kernel, run the test, and compare the numbers. Keep track of how many times
you had to send the agent back.

```{tip}
Ask yourself the questions you've asked all day: did the agent get the GPU
memory management right? Did it use the right reduction pattern? Does the
launch configuration make sense? You verify; the agent assists.
```

---

### Reflection

Take a few minutes to look back at the day's Agent Practice steps:

1. When did the agent give you **correct, useful** help?
2. What did it get wrong, and which check caught each mistake -- the Slurm
   filter, your own measurements, the `MPI_Ssend` test, the 2048 row, the
   padding warning, your spec?
3. Read the repository's `QWEN.md` (`cat ../../QWEN.md`). What did you teach
   your agent today, and what would you put in a `QWEN.md` for your own next
   project?
4. Would you have caught those mistakes if you **didn't** understand the
   fundamentals (Slurm, OpenMP, MPI, HIP)?

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
