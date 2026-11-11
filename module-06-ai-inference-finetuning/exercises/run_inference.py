"""
run_inference.py -- LLM inference exercise

EXERCISE: Fill in the 3 TODOs to compare generation settings and run the
model on the GPU.

Usage (the batch script activates the shared Python environment):
  sbatch submit_inference.sh run_inference.py
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.float16
TOP_P = 0.9


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
    """Generate a response to each prompt, one prompt at a time.

    Returns the responses, the total number of generated tokens, and the
    total generation time in seconds. The time covers only model.generate(),
    not tokenizing or decoding."""
    responses = []
    total_tokens = 0
    total_time = 0.0
    for prompt in prompts:
        inputs = tokenizer(format_prompt(tokenizer, prompt), return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        synchronize(device)
        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,  # use max_new_tokens only (avoids a warning)
            do_sample=True,   # sample, so that temperature has an effect
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
        synchronize(device)
        total_time += time.perf_counter() - t0

        new_ids = outputs[0, input_len:]
        n_tokens = count_generated_tokens(new_ids, tokenizer.eos_token_id)
        responses.append(tokenizer.decode(new_ids[:n_tokens], skip_special_tokens=True).strip())
        total_tokens += n_tokens

    return responses, total_tokens, total_time


def main():
    print("=" * 60)
    print("  LLM Inference Exercise")
    print("=" * 60)

    # TODO 1: Choose the generation settings to compare.
    # - max_new_tokens: the most tokens to generate per response (try 64, 128, 256)
    # - temperatures: a list of sampling temperatures; every prompt runs at each
    #   one, so you can compare them side by side (try [0.1, 0.7, 1.5])
    #
    # max_new_tokens = ???
    # temperatures = [???]
    max_new_tokens = 128
    temperatures = [0.7]

    # TODO 2: Add your own prompts to this list!
    # Keep the existing ones and add 2-3 of your own.
    prompts = [
        "What is the difference between a CPU and a GPU?",
        "Explain MPI in simple terms.",
        # Add your own prompts below:
    ]

    # TODO 3: Run on the GPU.
    # Change "cpu" to "cuda". PyTorch built for AMD GPUs (ROCm) still calls
    # the GPU "cuda". Leave it as "cpu" for the optional CPU baseline.
    device = "cpu"  # Replace this!

    if device == "cuda" and not torch.cuda.is_available():
        sys.exit('ERROR: device is "cuda", but no GPU is available. '
                 "Submit this script with sbatch so it runs on a GPU node.")

    print(f"\nModel:     {MODEL_NAME}")
    print(f"Precision: {DTYPE}")
    print(f"Device:    {device}")
    if device == "cuda":
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU threads used by PyTorch: {torch.get_num_threads()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e9:.2f}B")
    if device == "cuda":
        print(f"Allocated GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"\nSettings: max_new_tokens={max_new_tokens}, temperatures={temperatures}, "
          f"top_p={TOP_P}, sampling on")

    # The first generate() call includes one-time setup, so run a short,
    # untimed warm-up first.
    generate_responses(model, tokenizer, prompts[:1], device, 8, temperatures[0])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    all_tokens = 0
    all_time = 0.0
    for temperature in temperatures:
        print(f"\n{'=' * 60}")
        print(f"  temperature={temperature}")
        print(f"{'=' * 60}")

        responses, n_tokens, gen_time = generate_responses(
            model, tokenizer, prompts, device, max_new_tokens, temperature
        )
        all_tokens += n_tokens
        all_time += gen_time

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
            print(f"Response: {response}")
        print(f"\n[temperature {temperature}: {n_tokens} generated tokens in "
              f"{gen_time:.2f} s = {n_tokens / gen_time:.1f} tokens/s]")

    print(f"\n{'=' * 60}")
    print(f"  All temperatures: {all_tokens} generated tokens in {all_time:.2f} s "
          f"of generation = {all_tokens / all_time:.1f} tokens/s")
    print(f"{'=' * 60}")
    if device == "cuda":
        print(f"Peak allocated GPU memory while generating: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
