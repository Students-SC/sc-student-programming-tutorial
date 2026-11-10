"""
batch_inference_solution.py -- Solution for the Agent Practice: batch your prompts

Generates responses to the same prompts twice -- one prompt at a time, then
all prompts in a single batched model.generate() call -- and reports the
throughput of each.

Run from the exercises directory:
  sbatch submit_inference.sh solutions/batch_inference_solution.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 128

PROMPTS = [
    "What is the difference between a CPU and a GPU?",
    "Explain MPI in simple terms.",
    "What is a neural network?",
    "Why do supercomputers use Linux?",
    "Write a short poem about debugging code.",
    "What is OpenMP used for?",
    "What does a batch scheduler do?",
    "What is fine-tuning in machine learning?",
]


def generate(model, tokenizer, prompts, device):
    """Generate responses for a list of prompts in ONE model.generate() call."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            max_length=None,
            do_sample=False,  # greedy, so both runs produce the same responses
            pad_token_id=tokenizer.pad_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Count each response's tokens up to and including its end-of-sequence
    # token; a batch keeps running until its longest response finishes.
    responses, new_tokens = [], 0
    for row in outputs[:, input_len:].tolist():
        n = row.index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in row else len(row)
        new_tokens += n
        responses.append(tokenizer.decode(row[:n], skip_special_tokens=True).strip())
    return responses, new_tokens, elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Prompts have different lengths, so a batch is padded to the longest one.
    # A text generation model continues from the END of its input, so the
    # padding must go on the left.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map=device,
    )

    # Warm-up, so the first timing doesn't include one-time GPU setup
    generate(model, tokenizer, PROMPTS[:1], device)

    # --- One prompt at a time ---
    one_tokens, one_time = 0, 0.0
    for prompt in PROMPTS:
        _, n, elapsed = generate(model, tokenizer, [prompt], device)
        one_tokens += n
        one_time += elapsed
    one_rate = one_tokens / one_time
    print(f"\nOne prompt at a time:   {one_tokens} tokens in {one_time:.2f}s "
          f"= {one_rate:.1f} tok/s")

    # --- All prompts in one batch ---
    responses, batch_tokens, batch_time = generate(model, tokenizer, PROMPTS, device)
    batch_rate = batch_tokens / batch_time
    print(f"All {len(PROMPTS)} prompts batched: {batch_tokens} tokens in "
          f"{batch_time:.2f}s = {batch_rate:.1f} tok/s")
    print(f"Speedup from batching:  {batch_rate / one_rate:.1f}x")

    for prompt, response in zip(PROMPTS, responses):
        print(f"\n--- \"{prompt}\" ---\n{response}")


if __name__ == "__main__":
    main()
