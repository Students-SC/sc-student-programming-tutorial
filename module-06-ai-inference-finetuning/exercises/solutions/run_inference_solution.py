"""
run_inference_solution.py -- Solution: LLM inference exercise
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def generate_response(model, tokenizer, prompt, device, max_new_tokens=128,
                      temperature=0.7):
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = outputs.shape[1] - input_len
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip(), new_tokens, elapsed, tokens_per_sec


def main():
    print("=" * 60)
    print("  LLM Inference Exercise -- Solution")
    print("=" * 60)

    # TODO 1: SOLVED
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B")
    if device == "cuda":
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # TODO 2: SOLVED
    max_new_tokens = 128
    temperature = 0.7

    # TODO 3: SOLVED -- custom prompts added
    prompts = [
        "What is the difference between a CPU and a GPU?",
        "Explain MPI in simple terms.",
        "What is a neural network?",
        "Why do supercomputers use Linux?",
        "Write a short poem about debugging code.",
    ]

    print(f"\n{'=' * 60}")
    print(f"  Generating responses")
    print(f"  max_new_tokens={max_new_tokens}, temperature={temperature}")
    print(f"{'=' * 60}")

    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(prompts):
        response, n_tokens, elapsed, tok_s = generate_response(
            model, tokenizer, prompt, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        total_tokens += n_tokens
        total_time += elapsed

        print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
        print(f"Response: {response}")
        print(f"[{n_tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s]")

    print(f"\n{'=' * 60}")
    print(f"  Summary: {total_tokens} tokens in {total_time:.2f}s "
          f"= {total_tokens/total_time:.1f} tok/s average")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
