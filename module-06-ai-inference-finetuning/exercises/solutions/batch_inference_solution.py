"""
batch_inference_solution.py -- Solution for the Agent Practice: batch your prompts

Generates responses to the same prompts in two modes -- one prompt at a time,
then all prompts in a single model.generate() call -- and compares their
throughput and memory. Both modes use greedy decoding (do_sample=False), the
same model, precision, prompts, and token limit, and are warmed up first.

Timing covers model.generate() only: not loading, tokenizing, or decoding.
Generated tokens are counted up to and including each response's first
end-of-sequence token; padding after it isn't counted.

Run from the exercises directory:
  sbatch submit_inference.sh solutions/batch_inference_solution.py
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.float16
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


def count_generated_tokens(new_ids, eos_token_id):
    """Count the tokens a response actually generated: up to and including
    the first end-of-sequence token. Anything after it is padding."""
    new_ids = new_ids.tolist()
    if eos_token_id in new_ids:
        return new_ids.index(eos_token_id) + 1
    return len(new_ids)


def format_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Generate responses for a list of prompts in ONE model.generate() call.
    Returns the responses, the generated-token count, and the generation time."""
    inputs = tokenizer([format_prompt(tokenizer, p) for p in prompts],
                       return_tensors="pt", padding=True).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,  # input_ids and the attention mask, which marks the padding
        max_new_tokens=max_new_tokens,
        max_length=None,
        do_sample=False,  # greedy, so both modes choose tokens the same way
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # With left padding, every row's generated tokens start at input_len
    responses, n_tokens = [], 0
    for row in outputs[:, input_len:]:
        n = count_generated_tokens(row, tokenizer.eos_token_id)
        n_tokens += n
        responses.append(tokenizer.decode(row[:n], skip_special_tokens=True).strip())
    return responses, n_tokens, elapsed


def run_sequential(model, tokenizer, prompts):
    responses, n_tokens, elapsed = [], 0, 0.0
    for prompt in prompts:
        r, n, t = generate_batch(model, tokenizer, [prompt], MAX_NEW_TOKENS)
        responses += r
        n_tokens += n
        elapsed += t
    return responses, n_tokens, elapsed


def run_batched(model, tokenizer, prompts):
    return generate_batch(model, tokenizer, prompts, MAX_NEW_TOKENS)


def measure(mode, model, tokenizer):
    """Warm up a mode, then time it and record its peak allocated memory."""
    run = run_sequential if mode == "Sequential" else run_batched
    run(model, tokenizer, PROMPTS[:2])  # warm-up, untimed
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    responses, n_tokens, elapsed = run(model, tokenizer, PROMPTS)
    peak = torch.cuda.max_memory_allocated() / 1e9
    return responses, n_tokens, elapsed, peak


def main():
    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU is available. Submit this script with sbatch.")
    print(f"Model: {MODEL_NAME}, precision: {DTYPE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompts: {len(PROMPTS)}, max_new_tokens: {MAX_NEW_TOKENS}, greedy decoding")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Prompts have different lengths, so a batch is padded to the longest one.
    # A text generation model continues from the END of its input, so the
    # padding must go on the left.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map="cuda")
    model.eval()
    print(f"Allocated GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    results = {mode: measure(mode, model, tokenizer) for mode in ("Sequential", "Batched")}

    print(f"\n{'Mode':<11} {'Prompts':>7} {'Generated tokens':>16} {'Generation time (s)':>19} "
          f"{'Tokens/s':>9} {'Peak allocated (GB)':>19}")
    for mode, (_, n_tokens, elapsed, peak) in results.items():
        print(f"{mode:<11} {len(PROMPTS):>7} {n_tokens:>16} {elapsed:>19.2f} "
              f"{n_tokens / elapsed:>9.1f} {peak:>19.2f}")

    seq, bat = results["Sequential"][0], results["Batched"][0]
    same = sum(a == b for a, b in zip(seq, bat))
    print(f"\n{same} of {len(PROMPTS)} responses are identical in both modes. Greedy "
          "decoding always picks the most likely token, but a batch does the float16 "
          "arithmetic in a slightly different order, so when two tokens are nearly "
          "tied the choice can flip, and the rest of the response then differs.")

    for prompt, s, b in zip(PROMPTS, seq, bat):
        print(f"\n--- \"{prompt}\" ---\nSequential: {s}")
        if b != s:
            print(f"Batched:    {b}")


if __name__ == "__main__":
    main()
