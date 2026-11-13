"""
token_stats_solution.py -- Solution for the Agent Practice: token statistics

Reports each prompt's actual generated-token count, plus minimum, maximum,
average, and total response lengths for every temperature.

Run from the exercises directory:
  sbatch submit_inference.sh solutions/token_stats_solution.py
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.float16
TOP_P = 0.9

PROMPTS = [
    "What is the difference between a CPU and a GPU?",
    "Explain MPI in simple terms.",
    "What is a neural network?",
    "Why do supercomputers use Linux?",
    "Write a short poem about debugging code.",
]

NUM_PROMPTS = len(PROMPTS)


def synchronize(device):
    """Wait for all queued GPU work to finish, so a wall-clock timer covers it."""
    if device == "cuda":
        torch.cuda.synchronize()


def count_generated_tokens(new_ids, eos_token_id):
    """Count the tokens a response actually generated: up to and including
    the first end-of-sequence token. Anything after it is padding."""
    new_ids = new_ids.tolist()
    if eos_token_id in new_ids:
        return new_ids.index(eos_token_id) + 1
    return len(new_ids)


def format_prompt(tokenizer, prompt):
    """Wrap a prompt in the chat format the model was trained on."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate_responses(model, tokenizer, prompts, device, max_new_tokens,
                       temperature):
    """Generate responses one at a time and retain each response's token count."""
    responses = []
    token_counts = []
    total_time = 0.0

    for prompt in prompts:
        inputs = tokenizer(
            format_prompt(tokenizer, prompt), return_tensors="pt"
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        synchronize(device)
        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
        synchronize(device)
        total_time += time.perf_counter() - t0

        new_ids = outputs[0, input_len:]
        n_tokens = count_generated_tokens(new_ids, tokenizer.eos_token_id)
        responses.append(
            tokenizer.decode(
                new_ids[:n_tokens], skip_special_tokens=True
            ).strip()
        )
        token_counts.append(n_tokens)

    return responses, token_counts, total_time


def main():
    print("=" * 60)
    print("  Per-Prompt Token Statistics -- Solution")
    print("=" * 60)

    device = "cuda"
    max_new_tokens = 128
    temperatures = [0.1, 0.7, 1.5]
    prompts = PROMPTS[:NUM_PROMPTS]

    if not torch.cuda.is_available():
        sys.exit('ERROR: device is "cuda", but no GPU is available. '
                 "Submit this script with sbatch so it runs on a GPU node.")

    print(f"\nModel:     {MODEL_NAME}")
    print(f"Precision: {DTYPE}")
    print(f"Device:    {device}")
    print(f"GPU:       {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=device
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print(f"Allocated GPU memory after loading: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"\nSettings: {len(prompts)} prompts, "
          f"max_new_tokens={max_new_tokens}, temperatures={temperatures}, "
          f"top_p={TOP_P}, sampling on")

    # Warm up one short generation before measuring.
    generate_responses(model, tokenizer, prompts[:1], device, 8, temperatures[0])
    torch.cuda.reset_peak_memory_stats()

    all_tokens = 0
    all_time = 0.0
    for temperature in temperatures:
        print(f"\n{'=' * 60}")
        print(f"  temperature={temperature}")
        print(f"{'=' * 60}")

        responses, token_counts, gen_time = generate_responses(
            model, tokenizer, prompts, device, max_new_tokens, temperature
        )
        total_tokens = sum(token_counts)
        all_tokens += total_tokens
        all_time += gen_time

        for i, (prompt, response, n_tokens) in enumerate(
            zip(prompts, responses, token_counts)
        ):
            print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
            print(f"Response: {response}")
            print(f"Generated tokens: {n_tokens}")

        print(f"\nToken statistics: min={min(token_counts)}, "
              f"max={max(token_counts)}, "
              f"average={total_tokens / len(token_counts):.1f}, "
              f"total={total_tokens}")
        print(f"[temperature {temperature}: {total_tokens} generated tokens in "
              f"{gen_time:.2f} s = {total_tokens / gen_time:.1f} tokens/s]")

    print(f"\n{'=' * 60}")
    print(f"  All temperatures: {all_tokens} generated tokens in {all_time:.2f} s "
          f"of generation = {all_tokens / all_time:.1f} tokens/s")
    print(f"{'=' * 60}")
    print(f"Peak allocated GPU memory while generating: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
