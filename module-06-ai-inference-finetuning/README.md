# Module 6 -- AI on HPC: Inference & Fine-Tuning

> **Time:** 2:00 to 3:00 PM CST · 60 min total · ~20 min lecture · ~40 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain the difference between training, fine-tuning, and inference
- Run inference with a pre-trained language model on the MI210 GPU
- Fine-tune a text classification model using LoRA (parameter-efficient fine-tuning)
- Reason about GPU memory usage during inference vs. training

---

## Key Concepts

### What is AI/ML?

Machine learning is about finding patterns in data. Instead of writing explicit
rules, you show the computer many examples and it learns the rules itself.

```
  Traditional Programming:      Machine Learning:
  Rules + Data → Answers        Data + Answers → Rules (Model)
```

A **neural network** is a mathematical function with millions (or billions) of
adjustable numbers called **weights**. Training adjusts these weights so the
function produces good outputs for given inputs.

### Neural Networks in a Nutshell

```
  Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output
              ↑            ↑                ↑
           weights      weights          weights
```

- **Forward pass**: data flows through layers to produce a prediction
- **Loss**: measures how wrong the prediction is
- **Backpropagation**: computes how to adjust each weight to reduce the loss
- **Gradient descent**: actually updates the weights

### Inference vs. Training vs. Fine-Tuning

| | Inference | Training | Fine-Tuning |
|---|---|---|---|
| What | Use a trained model to make predictions | Train a model from scratch | Adapt a pre-trained model to a new task |
| Weights | Frozen (no changes) | Random → learned | Pre-trained → adjusted |
| Data needed | Just the input | Massive dataset | Small task-specific dataset |
| Compute cost | Low | Very high | Moderate |
| GPU memory | Model weights only | Weights + gradients + optimizer states | Less than full training |

### Parameter-Efficient Fine-Tuning (LoRA)

Full fine-tuning updates **every** weight in the model, which requires enormous
memory. **LoRA (Low-Rank Adaptation)** freezes the original weights and adds
small trainable "adapter" matrices:

```
  Original weight matrix W (frozen)
  +
  Low-rank adapter:  A × B  (trainable, much smaller)
  =
  Effective weight: W + A × B
```

The DistilBERT model you'll fine-tune has about 67 million parameters. With
this exercise's LoRA settings, only about 0.74 million (1.1%) are trained --
about 90x fewer. That is why fine-tuning fits easily on one GPU and finishes
in seconds.

### Why GPUs for AI?

Neural networks are mostly **matrix multiplications** -- exactly the kind of
massively parallel operation GPUs excel at:

- The MI210 has **6656 stream processors** running in parallel
- **64 GB HBM2e** with **1.6 TB/s bandwidth** for fast data access
- Generating text for one prompt at a time uses only a sliver of that
  parallelism, so the GPU may be only slightly faster than the CPU (you can
  measure this in Exercise 3). GPUs pull far ahead when they process many
  prompts together, which is called **batching** (Challenge C). Batching is
  how the shared AI server in Module 7 answers a whole classroom at once.

---

## Hands-On Exercises (~40 min)

First, navigate to the exercises directory for this module:

```bash
cd module-06-ai-inference-finetuning/exercises
```

```{important}
All exercises run as **batch jobs** that produce output files. Submit your
script, wait for it to complete, then examine the output while others use the
nodes.
```

### Exercise 1: Meet the Shared Python Environment

The AI exercises use Python packages such as PyTorch (built for AMD GPUs with
ROCm), Hugging Face `transformers` and `datasets`, and `peft`. Together they
take 16 GB, so your instructors installed them once in a shared, read-only
Python environment (a **virtual environment**, or venv) that everyone uses.
Each batch script in this module activates it for you, and you already tested
it with the GPU check in Getting Started.

#### Step 1: Look Inside

On the login node, activate the environment and list a few of its packages:

```bash
source "${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}/sc26_venv/bin/activate"
pip list 2>/dev/null | grep -iE '^(torch|transformers|datasets|peft) '
deactivate
```

```{note}
The shared environment is read-only, so `pip install` won't work in it.
Everything the exercises need is already installed. If a script reports that
the environment is missing, ask an instructor.
```

---

### Exercise 2: Run Your First Inference

#### Step 1: Examine the Example

Look at the simple inference example:

```bash
cat ../examples/simple_inference.py
```

#### Step 2: Review the Inference Workflow

This script loads a small pre-trained text generation model and generates
responses to a few prompts. Examine how it:
1. Loads the model and tokenizer
2. Moves the model to the GPU
3. Tokenizes input text
4. Generates output tokens
5. Decodes back to text

#### Step 3: Submit the Batch Job

```bash
sbatch submit_inference.sh
```

With no argument, `submit_inference.sh` runs this example. In Exercise 3,
you'll give it the name of your own script.

#### Step 4: Examine the Output

Check the output when it finishes:

```bash
cat inference_<JOBID>.out
```

**Questions:**
- How fast does the model generate text (tokens/sec)?
- How much GPU memory does the model use after loading, and at its peak while
  generating?
- Do the generated responses make sense?

---

### Exercise 3: Experiment with Inference

#### Step 1: Open the Exercise

Open the inference exercise in an editor on the login node:

```bash
vim run_inference.py
```

In Vim, press `i` to enter insert mode and make your edits. Press `Esc`, type
`:wq`, and press `Enter` to save and exit. You can instead use `nano`, `emacs`,
or an existing VS Code Remote SSH connection.

#### Step 2: (Optional) Try the CPU First

Until you complete TODO 1, the template runs the model on the CPU
(`device = "cpu"`). To get a baseline, submit it once before editing anything:

```bash
sbatch submit_inference.sh run_inference.py
```

Note the tokens/sec it reports, so you can compare with the GPU in Step 5.

#### Step 3: Complete the TODOs

There are **3 TODOs**:

1. **TODO 1**: Load the model onto the GPU (choose the right device)
2. **TODO 2**: Choose the generation settings: `max_new_tokens`, and a list of
   `temperatures` to compare (try `[0.1, 0.7, 1.5]`)
3. **TODO 3**: Add your own custom prompts to the prompt list

#### Step 4: Submit the Batch Job

Edit the file, add your own prompts, and submit it by naming the script after
the batch file:

```bash
sbatch submit_inference.sh run_inference.py
```

#### Step 5: Compare Temperatures

Your script runs every prompt at each temperature in `temperatures`, so one
job gives you a side-by-side comparison. With the suggested values, expect:
- `0.1` -- very focused, repetitive
- `0.7` -- balanced creativity
- `1.5` -- wild and unpredictable

Which temperature gives the most useful answers to your prompts? If you ran
the CPU baseline in Step 2, how many times faster is the GPU?

---

### Exercise 4: Fine-Tune a Model with LoRA

Now let's adapt a pre-trained model to a new task. We'll fine-tune a text
classification model (DistilBERT) on a sentiment analysis dataset using LoRA.

#### Step 1: Open the Fine-Tuning Script

Open the script in an editor, as in Exercise 3:

```bash
vim finetune_lora.py
```

#### Step 2: Complete the TODOs

There are **3 TODOs**:

1. **TODO 1**: Configure the LoRA adapter parameters
2. **TODO 2**: Write the training loop (forward pass, loss, backward, optimizer step)
3. **TODO 3**: Run evaluation and print accuracy

#### Step 3: Submit the Fine-Tuning Job

```bash
sbatch submit_finetune.sh
```

#### Step 4: Examine the Output

```bash
cat finetune_<JOBID>.out
```

**Questions:**
- What accuracy does the model achieve before fine-tuning (random baseline)?
- What accuracy after fine-tuning?
- How long did training take?
- How many parameters were actually trained (vs. total)?
- What was the peak GPU memory during training? Training stores gradients and
  optimizer state as well as the model, so why is the number still small?

---

### Challenge A: LoRA Hyperparameters

Experiment with different LoRA settings in `finetune_lora.py`:

- `r=4` (rank 4) vs. `r=16` (rank 16) vs. `r=64` (rank 64)
- `lora_alpha=16` vs. `lora_alpha=32`
- `num_epochs=1` vs. `num_epochs=3` vs. `num_epochs=5`

How do these affect final accuracy and training time?

### Challenge B: LoRA vs. Full Fine-Tuning

How much does LoRA actually save? Fine-tune the same model without it:

1. Copy your completed script: `cp finetune_lora.py finetune_full.py`
2. In `finetune_full.py`, delete the line that wraps the model with
   `get_peft_model`, so that every weight is trained.
3. Submit it: `sbatch submit_finetune.sh finetune_full.py`

Compare the two runs: trainable parameters, peak GPU memory during training,
training time, and accuracy after fine-tuning. How do both peaks compare with
the peak memory for inference in Exercise 2?

Then change `LEARNING_RATE` in `finetune_full.py` from `2e-4` to `2e-5` and
run it again. Why does training *all* of the weights need a smaller learning
rate than training only the LoRA adapters? (Hint: think about how far each
update moves the pre-trained weights.)

A full fine-tuning solution with the smaller learning rate is in
`solutions/finetune_full_solution.py`.

### Challenge C: Batch Your Prompts

So far the model answers one prompt at a time. Each step of generation reads
all of the model's weights from GPU memory, so the GPU does much more useful
work per step when it generates for many prompts together.

Modify a copy of your `run_inference.py` so it generates responses for all of
your prompts in a **single** `model.generate()` call:

1. Set `tokenizer.padding_side = "left"`. Prompts have different lengths, so
   they are padded to the same length, and a text generation model needs the
   padding on the left because it continues from the end of its input.
2. Format every prompt with the chat template, then tokenize the whole list
   at once: `tokenizer(texts, return_tensors="pt", padding=True)`.
3. Call `model.generate()` once, then decode each row of the output.

Compare the total tokens per second with one-prompt-at-a-time generation. A
solution that measures both is in `solutions/batch_inference_solution.py`:

```bash
sbatch submit_inference.sh solutions/batch_inference_solution.py
```

**Questions:**
- How much higher is the batched throughput? Did each individual response
  arrive faster or slower?
- The shared AI server in Module 7 answers the whole class at once. How does
  batching help it?

---

## Quick Reference

| Concept | Description |
|---------|-------------|
| **Inference** | Running a trained model on new inputs (forward pass only) |
| **Training** | Learning weights from scratch on a large dataset |
| **Fine-tuning** | Adapting a pre-trained model to a specific task |
| **LoRA** | Parameter-efficient fine-tuning using low-rank adapters |
| **Tokenizer** | Converts text to/from numerical tokens the model understands |
| **Temperature** | Controls randomness when sampling: lower is more focused, higher (>1) more random. Must be above 0; for deterministic output, turn sampling off with `do_sample=False` |

| PyTorch / HuggingFace | Purpose |
|-----------------------|---------|
| `AutoModelForCausalLM.from_pretrained(name)` | Load a pre-trained text generation model |
| `AutoTokenizer.from_pretrained(name)` | Load the matching tokenizer |
| `model.to("cuda")` | Move model to GPU |
| `model.generate(input_ids, ...)` | Generate text |
| `get_peft_model(model, lora_config)` | Wrap model with LoRA adapters |

---

**Next up:** [Module 7 -- AI Agents & Capstone](../module-07-ai-agents/README.md)
