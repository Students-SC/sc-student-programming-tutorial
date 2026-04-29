"""
finetune_lora.py -- Fine-tune a text classifier with LoRA

EXERCISE: Fill in the 3 TODOs to fine-tune DistilBERT for sentiment analysis.

We use the IMDB dataset (positive/negative movie reviews) and LoRA to
efficiently adapt the model with very few trainable parameters.

Usage (inside a Slurm job):
  source /work1/<group>/<username>/sc26_venv/bin/activate
  python3 finetune_lora.py
"""

import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
MAX_LENGTH = 256
TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 500


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def evaluate(model, eval_loader, device):
    """Compute accuracy on the evaluation set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load dataset ---
    print("\nLoading IMDB dataset...")
    dataset = load_dataset("imdb")
    train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    eval_data = dataset["test"].shuffle(seed=42).select(range(EVAL_SAMPLES))

    # --- Load model and tokenizer ---
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Apply LoRA ---
    # TODO 1: Configure the LoRA adapter.
    #
    # Create a LoraConfig with these settings:
    #   - task_type=TaskType.SEQ_CLS  (sequence classification)
    #   - r=8                          (rank of the low-rank matrices)
    #   - lora_alpha=16                (scaling factor)
    #   - lora_dropout=0.1             (dropout for regularization)
    #   - target_modules=["q_lin", "v_lin"]  (which layers to adapt)
    #
    # Then wrap the model: model = get_peft_model(model, lora_config)
    #
    # YOUR CODE HERE:
    # lora_config = LoraConfig(...)
    # model = get_peft_model(model, lora_config)

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    if device == "cuda":
        print(f"GPU memory after model load: "
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
    pre_accuracy = evaluate(model, eval_loader, device)
    print(f"\nAccuracy BEFORE fine-tuning: {pre_accuracy:.1%}")

    # --- Training loop ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    print(f"\nTraining for {NUM_EPOCHS} epochs on {TRAIN_SAMPLES} samples...")
    t_train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # TODO 2: Write the training step.
            #
            # Steps:
            #   1. Zero the gradients:  optimizer.zero_grad()
            #   2. Forward pass:        outputs = model(input_ids=input_ids,
            #                                           attention_mask=attention_mask,
            #                                           labels=labels)
            #   3. Get the loss:        loss = outputs.loss
            #   4. Backward pass:       loss.backward()
            #   5. Update weights:      optimizer.step()
            #
            # YOUR CODE HERE:
            loss = torch.tensor(0.0)  # Replace with real training step!

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: avg_loss = {avg_loss:.4f}")

    train_time = time.time() - t_train_start
    print(f"Training completed in {train_time:.1f}s")

    # TODO 3: Evaluate after fine-tuning and print the accuracy.
    #
    # Call the evaluate() function and print the result.
    #
    # YOUR CODE HERE:
    # post_accuracy = evaluate(model, eval_loader, device)
    # print(f"\nAccuracy AFTER fine-tuning: {post_accuracy:.1%}")
    # print(f"Improvement: {post_accuracy - pre_accuracy:+.1%}")

    print(f"\n{'=' * 60}")
    print("  Fine-tuning complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
