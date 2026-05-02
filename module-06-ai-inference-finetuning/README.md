# Module 6 -- AI on HPC: Inference & Fine-Tuning

<div class="event-meta module-meta">
  <span><strong>2:00 to 3:00 PM CST</strong></span>
  <span>60 min total</span>
  <span>~20 min lecture · ~40 min hands-on</span>
</div>

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

For a model with 110M parameters, LoRA might only train ~0.5M parameters --
a 200x reduction. This means fine-tuning fits on a single GPU and runs fast.

### Why GPUs for AI?

Neural networks are mostly **matrix multiplications** -- exactly the kind of
massively parallel operation GPUs excel at:

- The MI210 has **6656 stream processors** running in parallel
- **64 GB HBM2e** with **1.6 TB/s bandwidth** for fast data access
- This is why GPUs make inference 10-100x faster than CPUs

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

### Step 0: Verify Your Python Environment

You should already have the venv from Getting Started Step 4 (checked again in
Module 2). Verify it before starting the AI exercises:

```bash
source "$WORK/sc26_venv/bin/activate"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, ROCm: {torch.cuda.is_available()}')"
```

If the venv is missing, set it up now from this module's `exercises` directory:

```bash
sbatch ../../setup/setup_venv.sh
# Wait for the job to finish, then activate it:
source "$WORK/sc26_venv/bin/activate"
```

```{note}
The venv lives under `$WORK` (your `/work1/<project>/<username>/` directory)
because `/work1` has much more storage space than `$HOME`. The `$WORK`
environment variable is set automatically when you log in.
```

---

### Exercise 1: Your First Inference (Core)

Look at the simple inference example:

```bash
cat ../examples/simple_inference.py
```

This script loads a small pre-trained text generation model and generates
responses to a few prompts. Examine how it:
1. Loads the model and tokenizer
2. Moves the model to the GPU
3. Tokenizes input text
4. Generates output tokens
5. Decodes back to text

Submit it as a batch job:

```bash
sbatch submit_inference.sh
```

Check the output when it finishes:

```bash
cat inference_<JOBID>.out
```

**Questions:**
- How fast does the model generate text (tokens/sec)?
- How much GPU memory does the model use?
- Do the generated responses make sense?

---

### Exercise 2: Experiment with Inference (Core)

Open the inference exercise:

```bash
cat run_inference.py
```

There are **3 TODOs**:

1. **TODO 1**: Load the model onto the GPU (choose the right device)
2. **TODO 2**: Set generation parameters (max tokens, temperature)
3. **TODO 3**: Add your own custom prompts to the prompt list

Edit the file, add your own prompts, and submit:

```bash
sbatch submit_inference.sh
```

Try varying the `temperature` parameter:
- `temperature=0.1` -- very focused, repetitive
- `temperature=0.7` -- balanced creativity
- `temperature=1.5` -- wild and unpredictable

---

### Exercise 3: Fine-Tune a Model with LoRA (Core)

Now let's adapt a pre-trained model to a new task. We'll fine-tune a text
classification model (DistilBERT) on a sentiment analysis dataset using LoRA.

Look at the script:

```bash
cat finetune_lora.py
```

There are **3 TODOs**:

1. **TODO 1**: Configure the LoRA adapter parameters
2. **TODO 2**: Write the training loop (forward pass, loss, backward, optimizer step)
3. **TODO 3**: Run evaluation and print accuracy

Submit the fine-tuning job:

```bash
sbatch submit_finetune.sh
```

Check the output:

```bash
cat finetune_<JOBID>.out
```

**Questions:**
- What accuracy does the model achieve before fine-tuning (random baseline)?
- What accuracy after fine-tuning?
- How long did training take?
- How many parameters were actually trained (vs. total)?

---

### Challenge A: LoRA Hyperparameters

Experiment with different LoRA settings in `finetune_lora.py`:

- `r=4` (rank 4) vs. `r=16` (rank 16) vs. `r=64` (rank 64)
- `lora_alpha=16` vs. `lora_alpha=32`
- `num_epochs=1` vs. `num_epochs=3` vs. `num_epochs=5`

How do these affect final accuracy and training time?

### Challenge B: GPU Memory Profiling

Add `rocm-smi` monitoring to your batch script to see GPU memory usage during
inference vs. training:

```bash
# Add to your batch script before and during model loading:
rocm-smi --showmeminfo vram
# ... load model ...
rocm-smi --showmeminfo vram
```

How much GPU memory does the model use during inference vs. fine-tuning?

---

## Quick Reference

| Concept | Description |
|---------|-------------|
| **Inference** | Running a trained model on new inputs (forward pass only) |
| **Training** | Learning weights from scratch on a large dataset |
| **Fine-tuning** | Adapting a pre-trained model to a specific task |
| **LoRA** | Parameter-efficient fine-tuning using low-rank adapters |
| **Tokenizer** | Converts text to/from numerical tokens the model understands |
| **Temperature** | Controls randomness in text generation (0 = deterministic, >1 = creative) |

| PyTorch / HuggingFace | Purpose |
|-----------------------|---------|
| `AutoModelForCausalLM.from_pretrained(name)` | Load a pre-trained text generation model |
| `AutoTokenizer.from_pretrained(name)` | Load the matching tokenizer |
| `model.to("cuda")` | Move model to GPU |
| `model.generate(input_ids, ...)` | Generate text |
| `get_peft_model(model, lora_config)` | Wrap model with LoRA adapters |

---

**Next up:** [Module 7 -- AI Agents & Capstone](../module-07-ai-agents/README.md)
