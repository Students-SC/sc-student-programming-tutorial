# Module 6 -- AI on HPC: Inference & Fine-Tuning

> **Time:** 2:00 to 3:00 PM CST · 60 min total · ~20 min lecture · ~40 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain the difference between inference, pre-training, and fine-tuning
- Run inference with a pre-trained language model on the MI210 GPU and measure its throughput
- Fine-tune a text classification model using LoRA (parameter-efficient fine-tuning)
- Reason about GPU memory usage during inference and training
- Check an agent's change with measurements and the job's warnings

---

## Key Concepts

### What is Machine Learning?

Machine learning is about finding patterns in data. Instead of writing explicit
rules, you let the computer learn them from examples. In **supervised
learning**, the kind this module's fine-tuning exercise uses, each example
comes with the right answer (a label), and the computer learns a model that
maps inputs to answers:

```
  Traditional programming:      Supervised learning:
  Rules + Data → Answers        Data + Answers → Model
```

Not all machine learning needs labeled answers. A language model, for example,
is first trained on ordinary text by predicting the next word, so the text
itself provides the targets.

A **neural network** is a mathematical function with millions (or billions) of
adjustable numbers called **weights**, or **parameters**. Training adjusts
these weights so the function produces good outputs for given inputs.

### Neural Networks in a Nutshell

```
  Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output
              ↑            ↑                ↑
           weights      weights          weights
```

Each training step has four parts:

- **Forward pass**: data flows through the layers to produce predictions.
- **Loss**: a number that measures how far the predictions are from the
  training targets.
- **Backpropagation**: computes the **gradient** of the loss with respect to
  each trainable parameter, that is, which way and how strongly to nudge it to
  reduce the loss.
- **Optimizer step**: uses the gradients to update the trainable parameters.
  This module uses the **AdamW** optimizer, which keeps two running averages
  for every trainable parameter (its **optimizer state**) to choose the size of
  each update.

### Inference, Pre-training, and Fine-Tuning

Fine-tuning is a form of training: it continues training a model that has
already been pre-trained, to adapt it to a new task.

| | Inference | Pre-training | Fine-tuning |
|---|---|---|---|
| Purpose | Produce outputs with a trained model | Learn a base model, typically from a large dataset | Adapt a trained model to a task |
| Parameters updated | None | Typically all | All, or a selected subset |
| Data | The inputs to process | Training examples | Examples of the new task |
| GPU memory holds | Weights, intermediate results, workspace, and (for text generation) a cache of earlier tokens | Weights, saved intermediate results (activations), gradients, optimizer state, and workspace | The same kinds as pre-training; LoRA reduces the trainable parameters and their gradients and optimizer state |
| Compute required | Depends on the model, the inputs, and how many there are | Usually substantial | Depends on the model, the data, and the method |

### Two Models in This Module

This module uses two different models:

- **TinyLlama 1.1B Chat** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, 1.1 billion
  parameters) generates text. It predicts one **token** (a word or part of a
  word) at a time, each based on the tokens before it. It is a chat model: a
  base language model that was further trained to answer requests written in
  a chat format, which is why the scripts wrap each prompt in its **chat
  template**.
- **DistilBERT** (`distilbert-base-uncased`, 67 million parameters) reads a
  whole piece of text at once. You'll fine-tune it to classify movie reviews
  from the IMDB dataset as positive or negative, training on 2,000 reviews and
  evaluating on 500 others. DistilBERT was pre-trained on ordinary text; the
  classification layers on top of it are new, and start with random weights.

Fine-tuning the classifier does not change the text-generation model.

### Parameter-Efficient Fine-Tuning (LoRA)

Full fine-tuning updates **every** weight in the model, so it needs a gradient
and optimizer state for every one. **LoRA (Low-Rank Adaptation)** freezes
selected weight matrices and trains a small, low-rank update for each:

```
  Original weight matrix W (frozen, d × d)
  +
  Low-rank update: B × A, where B is d × r and A is r × d, and the rank r
                   is much smaller than d; the update is scaled by lora_alpha / r
  =
  Effective weight: W + (lora_alpha / r) × B × A
```

In this module's exercise, LoRA adapts two weight matrices (`q_lin` and
`v_lin`) in each of DistilBERT's 6 layers with rank r = 8, which adds 147,456
trainable parameters. The new classification layers are trained as well, and
they have 592,130 parameters. Together that is 739,586 trainable parameters out
of about 67 million (1.1%), about 90 times fewer than full fine-tuning.

Fewer trainable parameters means less memory for gradients and optimizer
state. But the frozen model must still be in GPU memory, and each training
step still runs the whole model forward and backward and stores its
intermediate results, so 90 times fewer trainable parameters does not mean 90
times less memory or time. This exercise uses a small model and a subset of
the dataset to keep training practical within the tutorial. LoRA reduces the
number of trainable parameters and their associated gradient and
optimizer-state memory.

### Why GPUs for AI?

Neural networks are mostly **matrix multiplications**, which split into many
independent multiply-adds that a GPU can run in parallel:

- The MI210 has **104 compute units** with **6656 stream processors**. A stream
  processor is a simple arithmetic lane, not an independent CPU-style core:
  lanes work in groups of 64 (a wavefront, as in Module 5) that run the same
  instruction on different data.
- **64 GB HBM2e** with **1.6 TB/s bandwidth** for fast data access.

Generating text has two phases. First the model processes the whole prompt at
once, which is a lot of parallel work. Then it generates one token per step,
and each step depends on the token before it. For a single prompt, each step
does a little arithmetic but has to move a large share of the model's weights
from memory, so it can use only a small part of the GPU's compute capacity and
is often limited by memory bandwidth instead.

**Batching**, generating for several prompts in the same steps, lets those
sequences share each pass of the model, which can greatly raise the total
number of tokens generated per second. How much depends on the model, the
sequence lengths, the batch size, and the software. A batch needs more memory,
and it doesn't make every individual response arrive sooner. The shared AI
server behind the tutorial's agent batches requests from the whole classroom.
Servers like it add requests to the batch and remove finished ones as they
go; this module's Agent Practice uses a simpler, fixed batch.

---

## Hands-On Exercises (~40 min)

Suggested pacing: about 5 minutes for Exercises 1 and 2, 8 minutes for
Exercise 3, 12 minutes for Exercise 4, and 15 minutes for the Agent Practice.
Exercise 3's CPU baseline and the challenges are optional.

First, navigate to the exercises directory for this module:

```bash
cd module-06-ai-inference-finetuning/exercises
```

> **Important**
>
> All exercises run as **batch jobs** that produce output files. Submit your
> script, wait for it to complete, then examine the output while others use the
> nodes. Never run these models on the login node.

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

> **Note**
>
> The shared environment is read-only, so `pip install` won't work in it.
> Everything the exercises need is already installed. If a script reports that
> the environment is missing, ask an instructor.

The models and the dataset these exercises use are pre-downloaded too, so your
jobs load them from a shared cache and never download anything. The way these
scripts load the dataset, the `datasets` library needs to write next to it,
and the shared copy is read-only, so the first fine-tuning job you run copies
the dataset (128 MB) into your `$WORK` directory.

---

### Exercise 2: Run Your First Inference

#### Step 1: Examine the Example

Look at the simple inference example:

```bash
cat ../examples/simple_inference.py
```

#### Step 2: Review the Inference Workflow

This script loads TinyLlama and generates responses to a few prompts. Examine
how it:
1. Loads the model and tokenizer, and moves the model to the GPU
2. Wraps each prompt in the chat template and tokenizes it
3. Runs a short, untimed warm-up generation
4. Generates output tokens, timing only `model.generate()`
5. Counts the tokens it actually generated and decodes them back to text

> **Note**
>
> PyTorch built for AMD GPUs (ROCm) still calls the GPU `"cuda"` and uses the
> `torch.cuda` functions, so the same code runs on AMD and NVIDIA GPUs.

GPU work runs in the background, so the script calls
`torch.cuda.synchronize()` before and after `model.generate()` to make sure the
timer covers all of the generation. It counts each response's tokens up to and
including the token that ends the response.

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
- How many tokens per second does the model generate?
- The script runs a warm-up generation before it starts timing. Why? (Think
  back to Module 5.)
- How much GPU memory is allocated after loading, and at the peak while
  generating? Besides the weights, what needs memory while generating?
- Do the generated responses make sense?

The scripts report memory **allocated** to PyTorch tensors. PyTorch's caching
allocator reserves somewhat more, and `rocm-smi` shows everything in use on the
GPU.

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

#### Step 2: Complete TODOs 1 and 2

There are **3 TODOs**. Do the first two now:

1. **TODO 1**: Choose the generation settings: `max_new_tokens`, and a list of
   `temperatures` to compare (try `[0.1, 0.7, 1.5]`)
2. **TODO 2**: Add your own prompts to the prompt list

#### Step 3: (Optional) Run on the CPU First

Until you complete TODO 3, the script runs the model on the CPU. For a
baseline, submit it once now:

```bash
sbatch submit_inference.sh run_inference.py
```

With the suggested settings, this job takes about 4 minutes. Note the tokens
per second on its `All temperatures` line, and the number of CPU threads it
used.

#### Step 4: Complete TODO 3 and Run on the GPU

3. **TODO 3**: Change the device from `"cpu"` to `"cuda"`

Then submit the script again:

```bash
sbatch submit_inference.sh run_inference.py
```

#### Step 5: Compare Temperatures

Your script runs every prompt at each temperature in `temperatures`, so one
job gives you a side-by-side comparison. Temperature only matters when the
model **samples** its next token at random from the likely ones
(`do_sample=True`, as here). In general:
- `0.1` concentrates the choice on the highest-scoring tokens, so responses
  are focused and can repeat themselves.
- `0.7` allows more variation.
- `1.5` flattens the choice, which often adds variation and mistakes. (The
  script's `top_p=0.9` still drops the least likely tokens, so the text
  usually stays readable.)

With `do_sample=False`, the model instead always picks the most likely token,
which is called **greedy decoding**. A lower temperature doesn't make an answer
more correct, and a model this small can give weak answers at every
temperature.

**Questions:**
- Which temperature gives the most useful answers to your prompts?
- If you ran the CPU baseline, compare the two runs' `All temperatures` lines.
  They did the same work, so how many times faster is the GPU?

---

### Exercise 4: Fine-Tune a Model with LoRA

Now let's adapt a pre-trained model to a new task. We'll fine-tune a text
classification model (DistilBERT) on the IMDB sentiment dataset using LoRA.

#### Step 1: Open the Fine-Tuning Script

Open the script in an editor, as in Exercise 3:

```bash
vim finetune_lora.py
```

#### Step 2: Complete the TODOs

There are **3 TODOs**:

1. **TODO 1**: Configure the LoRA adapter parameters
2. **TODO 2**: Write the training step (clear the gradients, forward pass,
   loss, backward pass, optimizer step)
3. **TODO 3**: Evaluate the model after training and print its accuracy

#### Step 3: Submit the Fine-Tuning Job

```bash
sbatch submit_finetune.sh
```

#### Step 4: Examine the Output

```bash
cat finetune_<JOBID>.out
```

> **Note**
>
> `finetune_<JOBID>.err` contains a "LOAD REPORT" that lists `MISSING`
> classifier weights and `UNEXPECTED` vocabulary weights. That's expected: the
> checkpoint was pre-trained to predict words, not classify reviews, so the
> classification layers are new, and training them is the point of this
> exercise.

The script fixes its random seed, so running it again gives the same data
selection and starting weights.

**Questions:**
- Compare the accuracy before fine-tuning with the majority-class baseline
  (always guessing the more common label). The new classification layers
  start with random weights, so the starting accuracy is close to a guess, but
  not exactly 50%.
- What accuracy does the model reach after fine-tuning?
- How long did the training loop take?
- How many parameters were trained, and how many of those are LoRA adapters?
- What was the peak allocated GPU memory during training? Training stores
  gradients and optimizer state for the trained parameters, and intermediate
  results for the whole model. Which of those does LoRA shrink?

---

### Agent Practice: Batch Your Prompts

So far the model answers one prompt at a time. Each step of generation reads
all of the model's weights from GPU memory, so the GPU does much more useful
work per step when it generates for many prompts together. Have the agent make
that change, and check it with measurements and the job's warnings.

#### Step 1: Ask for the Change

Copy your completed script from Exercise 3, then start Qwen Code:

```bash
cp run_inference.py batch_agent.py
bash ../../setup/launch_qwen.sh
```

Ask it:

> Change batch_agent.py so it generates the responses for all prompts at each temperature in a single model.generate() call, and prints the total tokens per second.

Read the change before you approve it. Predict: will the total tokens per
second go up or down? Will each individual response arrive sooner or later?

#### Step 2: Run It and Check

```bash
sbatch submit_inference.sh batch_agent.py
```

When the job finishes, check three things:
- Is the total tokens per second higher than in Exercise 3?
- Do the responses still make sense?
- Does `inference_<JOBID>.err` contain a warning about padding?

If something is wrong, give the agent the warning or the output and ask it to
fix the script.

**Questions:**
- Prompts have different lengths, so they are padded to the same length.
  Which side did the agent pad on? Why does a text generation model need the
  padding on the left?
- The shared AI server behind Qwen Code answers the whole class at once. How
  does batching help it?
- Type `/stats model` in Qwen Code. How many tokens has your session used?
  Every request behind those numbers was an inference job like yours.

A solution that measures both ways is in
`solutions/batch_inference_solution.py`.

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
