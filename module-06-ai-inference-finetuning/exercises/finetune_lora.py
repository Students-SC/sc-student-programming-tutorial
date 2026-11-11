"""
finetune_lora.py -- Fine-tune a text classifier with LoRA

EXERCISE: Read, explain, and enable the 3 commented-out code blocks (TODOs 1-3)
to fine-tune DistilBERT for sentiment analysis. To enable a block, delete the
first "#" of each line between its BEGIN and END markers, then set its flag
below to True.

We use the IMDB dataset (positive/negative movie reviews) and LoRA to
efficiently adapt the model with very few trainable parameters.

Usage (the batch script activates the shared Python environment):
  sbatch submit_finetune.sh
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

# Set each flag to True after you have enabled and reviewed its code block.
# These are reminders that the exercise is finished, not tests of your code.
LORA_CONFIG_READY = False    # TODO 1
TRAINING_STEP_READY = False  # TODO 2
EVALUATION_READY = False     # TODO 3


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
    print("  LoRA Fine-Tuning Exercise")
    print("=" * 60)

    unfinished = [name for name, ready in [
        ("TODO 1 (LoRA configuration)", LORA_CONFIG_READY),
        ("TODO 2 (training step)", TRAINING_STEP_READY),
        ("TODO 3 (evaluation)", EVALUATION_READY),
    ] if not ready]
    if unfinished:
        sys.exit("Not finished yet: " + ", ".join(unfinished) + ".\n"
                 "Enable each code block (delete the first # of each line between its "
                 "BEGIN and END markers), then set its flag at the top of the file to True.")

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

    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")

    # --- Apply LoRA ---
    # TODO 1: Read the configuration, then enable it. Be ready to explain r and
    # target_modules (see "Configuring LoRA" in the README).
    # ----- BEGIN TODO 1 -----
#    lora_config = LoraConfig(
#        task_type=TaskType.SEQ_CLS,         # sequence classification
#        r=8,                                # rank of the low-rank update
#        lora_alpha=16,                      # scales the update by lora_alpha / r = 2
#        lora_dropout=0.1,                   # dropout on the adapters' input, in training
#        target_modules=["q_lin", "v_lin"],  # attention query and value layers
#    )
#    model = get_peft_model(model, lora_config)
    # ----- END TODO 1 -----

    total_params = sum(p.numel() for p in model.parameters())
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    trainable_params = sum(trainable.values())
    lora_params = sum(v for n, v in trainable.items() if "lora_" in n)
    print(f"Parameters after wrapping with LoRA: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.2f}% of {total_params:,})")
    print(f"  LoRA adapters: {lora_params:,}, other trainable layers: "
          f"{trainable_params - lora_params:,}")
    if lora_params == 0:
        sys.exit("ERROR: no LoRA parameters are trainable, so this would train every "
                 "weight. Did you enable the TODO 1 block?\n"
                 "(For Challenge B's full fine-tuning, delete this check.)")

    model.to(device)
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

            # TODO 2: Read the training step, then enable it (see "One Training
            # Step" in the README). Lines that are still comments afterward are
            # explanations.
            # ----- BEGIN TODO 2 -----
#            # Discard gradients from the previous batch.
#            optimizer.zero_grad()
#
#            # Predict the labels and compute the classification loss.
#            outputs = model(
#                input_ids=input_ids,
#                attention_mask=attention_mask,
#                labels=labels,
#            )
#            loss = outputs.loss
#
#            # Compute gradients for the trainable parameters.
#            loss.backward()
#
#            # Update those parameters using the gradients.
#            optimizer.step()
            # ----- END TODO 2 -----

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: avg_loss = {avg_loss:.4f}")

    torch.cuda.synchronize()
    train_time = time.perf_counter() - t_train_start
    print(f"Training loop completed in {train_time:.1f} s (excludes evaluation)")
    print(f"Peak allocated GPU memory during training: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # TODO 3: Enable the evaluation after fine-tuning. evaluate() is defined at
    # the top of this file; you don't need to write it.
    # ----- BEGIN TODO 3 -----
#    post_accuracy = evaluate(model, eval_loader, device)
#    print(f"\nAccuracy after fine-tuning: {post_accuracy:.1%}")
#    print(f"Change: {post_accuracy - pre_accuracy:+.1%}")
    # ----- END TODO 3 -----

    print(f"\n{'=' * 60}")
    print("  Fine-tuning complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
