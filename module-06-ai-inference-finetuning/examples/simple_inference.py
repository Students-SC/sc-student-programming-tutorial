"""
simple_inference.py -- Run inference with a pre-trained text generation model

This script demonstrates the basic pattern for LLM inference on a GPU:
  1. Load a pre-trained model and tokenizer
  2. Move the model to the GPU
  3. Tokenize prompts, generate responses, decode back to text

Usage (from the exercises directory; the batch script activates the shared
Python environment):
  sbatch submit_inference.sh
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9


def count_generated_tokens(new_ids, eos_token_id):
    """Count the tokens a response actually generated: up to and including
    the first end-of-sequence token. Anything after it is padding."""
    new_ids = new_ids.tolist()
    if eos_token_id in new_ids:
        return new_ids.index(eos_token_id) + 1
    return len(new_ids)


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens):
    """Generate one response. Returns the text, the number of generated
    tokens, and the time spent in model.generate() alone."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # GPU work runs in the background, so wait for it before reading the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        max_length=None,  # use max_new_tokens only (avoids a warning)
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_ids = outputs[0, input_len:]
    n_tokens = count_generated_tokens(new_ids, tokenizer.eos_token_id)
    response = tokenizer.decode(new_ids[:n_tokens], skip_special_tokens=True)
    return response.strip(), n_tokens, elapsed


def main():
    print("=" * 60)
    print("  Simple LLM Inference Demo")
    print("=" * 60)

    # PyTorch built for AMD GPUs (ROCm) still calls the GPU "cuda"
    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU is available. Submit this script with sbatch "
                 "so it runs on a GPU node.")
    print(f"\nGPU:       {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading model: {MODEL_NAME}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda")
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f} s")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B, precision: {DTYPE}")
    print(f"Allocated GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompts = [
        "What is a supercomputer?",
        "Explain GPU programming in one sentence.",
        "Write a haiku about parallel computing.",
    ]

    # The first generate() call includes one-time GPU setup, so run a short,
    # untimed warm-up first, then start measuring peak memory.
    generate(model, tokenizer, prompts[0], 8)
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{'=' * 60}")
    print(f"  Generating responses ({len(prompts)} prompts)")
    print(f"  max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"{'=' * 60}")

    total_tokens = 0
    total_time = 0.0
    for i, prompt in enumerate(prompts):
        response, n_tokens, elapsed = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        total_tokens += n_tokens
        total_time += elapsed
        print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
        print(f"Response: {response}")
        print(f"[{n_tokens} tokens in {elapsed:.2f} s = {n_tokens / elapsed:.1f} tokens/s]")

    print(f"\nAll prompts: {total_tokens} generated tokens in {total_time:.2f} s "
          f"of generation = {total_tokens / total_time:.1f} tokens/s")
    print(f"Peak allocated GPU memory while generating: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
