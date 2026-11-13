"""
simple_inference.py -- Generate text with TinyLlama on a GPU.

The script loads the model once, warms up generation, and then processes
three prompts sequentially. It reports output tokens/s and allocated GPU memory.

Workflow:
  1. Load the tokenizer and model onto the GPU.
  2. Format each prompt as a chat and convert it to token IDs.
  3. Generate new tokens, then decode only the continuation.

Usage (from the exercises directory):
  sbatch submit_inference.sh

The batch script activates the shared Python environment and configures the
pre-downloaded cache. Do not run this model on the login node.
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.float16  # Store model weights in FP16.

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9


def count_generated_tokens(new_ids, eos_token_id):
    """Count output tokens through the first end-of-sequence token, inclusive.

    If no end-of-sequence token occurs, count the whole continuation.
    Ignore any IDs after the first end-of-sequence token, such as batch padding.
    """
    new_ids = new_ids.tolist()
    if eos_token_id in new_ids:
        return new_ids.index(eos_token_id) + 1
    return len(new_ids)


# Disable gradient tracking for every call to this helper.
@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens):
    """Return response text, actual output-token count, and generation seconds.

    Time only model.generate(), including prefill and decode. Exclude chat
    formatting, tokenization, input transfer, token counting, and decoding.
    """
    # Add the conversation markers this chat model expects, including the
    # assistant-response prefix. This formatting does not generate a reply.
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Tokenize into input_ids and attention_mask; put them on the model's GPU.
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # GPU work is asynchronous. Finish earlier work before starting the timer.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,  # Expand input_ids and attention_mask into named arguments.
        max_new_tokens=max_new_tokens,  # Upper limit; EOS can end a reply sooner.
        max_length=None,  # Use the new-token limit in this tested configuration.
        do_sample=True,  # Sample tokens; responses can differ between runs.
        temperature=TEMPERATURE,  # Reshape probabilities before sampling.
        top_p=TOP_P,  # Restrict sampling to the most likely cumulative mass.
        pad_token_id=tokenizer.eos_token_id,  # Use EOS for padding too.
    )
    torch.cuda.synchronize()  # Include completion of all generation GPU work.
    elapsed = time.perf_counter() - t0

    # Decoder-only output contains the input followed by its continuation.
    # Remove the input; count EOS if present, but hide special tokens in text.
    new_ids = outputs[0, input_len:]
    n_tokens = count_generated_tokens(new_ids, tokenizer.eos_token_id)
    response = tokenizer.decode(new_ids[:n_tokens], skip_special_tokens=True)
    return response.strip(), n_tokens, elapsed


def main():
    print("=" * 60)
    print("  Simple LLM Inference Demo")
    print("=" * 60)

    # ROCm PyTorch uses "cuda" and torch.cuda for AMD GPUs too.
    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU is available. Submit this script with sbatch "
                 "so it runs on a GPU node.")
    print(f"\nGPU:       {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading model: {MODEL_NAME}")
    t0 = time.perf_counter()
    # Loading and device placement happen once, outside generation timing.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda")
    # Evaluation mode disables dropout. The decorator on generate() separately
    # disables gradient tracking; neither call updates the model's parameters.
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f} s")

    # Report memory occupied by PyTorch tensors, not all reserved GPU memory.
    # Values are decimal GB (1 GB = 1e9 bytes).
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B, precision: {DTYPE}")
    print(f"Allocated GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompts = [
        "What is a supercomputer?",
        "Explain GPU programming in one sentence.",
        "Write a haiku about parallel computing.",
    ]

    # Warm up one-time setup before the measured prompts. The helper times this
    # call too, but we discard its results and exclude them from the totals.
    # Reset the peak after warm-up; the model weights remain allocated.
    generate(model, tokenizer, prompts[0], 8)
    torch.cuda.reset_peak_memory_stats()

    print(f"\n{'=' * 60}")
    print(f"  Generating responses ({len(prompts)} prompts)")
    print(f"  max_new_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}, top_p={TOP_P}")
    print(f"{'=' * 60}")

    total_tokens = 0
    total_time = 0.0
    # Generate one prompt at a time.
    for i, prompt in enumerate(prompts):
        response, n_tokens, elapsed = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        total_tokens += n_tokens
        total_time += elapsed
        print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
        print(f"Response: {response}")
        print(f"[{n_tokens} tokens in {elapsed:.2f} s = {n_tokens / elapsed:.1f} tokens/s]")

    # Aggregate throughput is total output tokens divided by summed generation
    # time, not the average of the per-prompt rates or the full script runtime.
    print(f"\nAll prompts: {total_tokens} generated tokens in {total_time:.2f} s "
          f"of generation = {total_tokens / total_time:.1f} tokens/s")
    # This peak includes model weights, KV cache, and other PyTorch tensors.
    # It is not just the extra memory above the after-loading measurement.
    print(f"Peak allocated GPU memory while generating: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
