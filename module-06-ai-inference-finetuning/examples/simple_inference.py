"""
simple_inference.py -- Run inference with a pre-trained text generation model

This script demonstrates the basic pattern for LLM inference on a GPU:
  1. Load a pre-trained model and tokenizer
  2. Move the model to the GPU
  3. Tokenize prompts, generate responses, decode back to text

Usage (inside a Slurm job):
  source /work1/<group>/<username>/sc26_venv/bin/activate
  python3 simple_inference.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    print("=" * 60)
    print("  Simple LLM Inference Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B")

    if device == "cuda":
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {mem_used:.2f} GB")

    # Define prompts
    prompts = [
        "What is a supercomputer?",
        "Explain GPU programming in one sentence.",
        "Write a haiku about parallel computing.",
    ]

    print(f"\n{'=' * 60}")
    print(f"  Generating responses ({len(prompts)} prompts)")
    print(f"{'=' * 60}")

    for i, prompt in enumerate(prompts):
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = outputs.shape[1] - input_len
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
        print(f"Response: {response.strip()}")
        print(f"[{new_tokens} tokens in {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s]")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
