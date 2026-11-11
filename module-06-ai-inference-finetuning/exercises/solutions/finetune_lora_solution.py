"""
finetune_lora_solution.py -- Solution: LoRA fine-tuning for sentiment analysis

Run from the exercises directory:
  sbatch submit_finetune.sh solutions/finetune_lora_solution.py
"""

import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "distilbert-base-uncased"
DATASET = "imdb"
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
MAX_LENGTH = 256
TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 500
SEED = 42  # fixes the data selection and the new classifier's starting weights


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


@torch.inference_mode()
def evaluate(model, eval_loader, device):
    """Compute accuracy on the evaluation set."""
    model.eval()
    correct = 0
    total = 0
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 60)
    print("  LoRA Fine-Tuning -- Solution")
    print("=" * 60)

    # PyTorch built for AMD GPUs (ROCm) still calls the GPU "cuda"
    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU is available. Submit this script with sbatch "
                 "so it runs on a GPU node.")
    device = "cuda"
    torch.manual_seed(SEED)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_NAME}, dataset: {DATASET}, seed: {SEED}")
    print(f"Training examples: {TRAIN_SAMPLES} (from the train split), "
          f"evaluation examples: {EVAL_SAMPLES} (from the test split)")
    print(f"Epochs: {NUM_EPOCHS}, batch size: {BATCH_SIZE}, "
          f"max length: {MAX_LENGTH} tokens, learning rate: {LEARNING_RATE}")

    # --- Load dataset ---
    print("\nLoading IMDB dataset...")
    dataset = load_dataset(DATASET)
    train_data = dataset["train"].shuffle(seed=SEED).select(range(TRAIN_SAMPLES))
    eval_data = dataset["test"].shuffle(seed=SEED).select(range(EVAL_SAMPLES))

    # --- Load model and tokenizer ---
    # The pre-trained DistilBERT body is loaded from the checkpoint; the
    # classification layers on top are new, with random starting weights.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Apply LoRA ---
    # TODO 1: SOLVED
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)

    model.to(device)

    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    trainable_params = sum(trainable.values())
    lora_params = sum(v for n, v in trainable.items() if "lora_" in n)
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    print(f"  LoRA adapters: {lora_params:,}, other layers: {trainable_params - lora_params:,}")
    print(f"Allocated GPU memory after model load: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- Tokenize datasets ---
    print("\nTokenizing datasets...")
    train_encoded = train_data.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    eval_encoded = eval_data.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    train_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_encoded, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_encoded, batch_size=BATCH_SIZE)

    # --- Evaluate before fine-tuning ---
    eval_labels = list(eval_data["label"])
    majority = max(eval_labels.count(0), eval_labels.count(1)) / len(eval_labels)
    print(f"\nMajority-class baseline (always guess the most common label): {majority:.1%}")
    pre_accuracy = evaluate(model, eval_loader, device)
    print(f"Accuracy before fine-tuning: {pre_accuracy:.1%}")

    # --- Training loop ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    print(f"\nTraining for {NUM_EPOCHS} epochs on {TRAIN_SAMPLES} examples...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_train_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # TODO 2: SOLVED
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: avg_loss = {avg_loss:.4f}")

    torch.cuda.synchronize()
    train_time = time.perf_counter() - t_train_start
    print(f"Training loop completed in {train_time:.1f} s (excludes evaluation)")
    print(f"Peak allocated GPU memory during training: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # TODO 3: SOLVED
    post_accuracy = evaluate(model, eval_loader, device)
    print(f"\nAccuracy after fine-tuning: {post_accuracy:.1%}")
    print(f"Change: {post_accuracy - pre_accuracy:+.1%}")

    print(f"\n{'=' * 60}")
    print("  Fine-tuning complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
