"""
train.py – QLoRA fine-tuning of LLaVA-1.5-7B on ChartQA.

RNN/Transformer HW4: Multimodal AI – Visual Instruction Tuning
Requirements:
  - 4-bit quantisation via bitsandbytes  (VRAM ≤ 24 GB)
  - LoRA adapters on LLM attention layers only (q_proj, v_proj)
  - Vision encoder (ViT) weights FROZEN throughout
  - Training loss exported to CSV + TensorBoard
"""

import os
import csv
import json
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import (
    MODEL_ID, OUTPUT_DIR, MAX_SEQ_LENGTH,
    NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE,
    WARMUP_STEPS, LOGGING_STEPS, SAVE_STEPS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    DATASET_NAME, DATASET_SAMPLE_SIZE,
    BNB_LOAD_IN_4BIT, BNB_QUANT_TYPE, BNB_DOUBLE_QUANT, BNB_COMPUTE_DTYPE,
)
from load import load_train_dataset, get_answer, build_conversation


# ── Model loading ─────────────────────────────────────────────────────────────

def create_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=BNB_LOAD_IN_4BIT,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
    )


def load_model_and_processor():
    print(f"Loading {MODEL_ID} in 4-bit…")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=create_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    return model, processor


# ── QLoRA setup ───────────────────────────────────────────────────────────────

def apply_qlora(model):
    """
    1. prepare_model_for_kbit_training  – freeze all weights, cast norms to fp32
    2. Add LoRA adapters to LLM attention layers only
    3. Re-freeze vision tower (belt-and-suspenders guarantee)
    """
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Required for gradient checkpointing + PEFT compatibility
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # CRITICAL: ensure vision tower is NOT trainable (assignment requirement)
    frozen_vision = 0
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False
            frozen_vision += param.numel()
    print(f"Vision tower frozen: {frozen_vision:,} params")
    model.print_trainable_parameters()
    return model


# ── Multimodal data collator ───────────────────────────────────────────────────

class LLaVAMultimodalCollator:
    """
    For each sample:
      1. Build full conversation text (question + answer)
      2. Build prompt-only text  → determines where answer tokens start
      3. Set labels = input_ids, then mask prompt positions with -100
         so the model only learns to predict the answer.
    """

    def __init__(self, processor, max_length: int = MAX_SEQ_LENGTH):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples):
        batch_input_ids      = []
        batch_attention_mask = []
        batch_pixel_values   = []
        batch_labels         = []

        for ex in examples:
            query  = ex["query"]
            answer = get_answer(ex["label"])
            image  = ex["image"].convert("RGB")

            # ── Full conversation: USER + ASSISTANT ──
            full_text = self.processor.apply_chat_template(
                build_conversation(query, answer),
                add_generation_prompt=False,
            )

            # ── Prompt only: USER (to locate where answer begins) ──
            prompt_text = self.processor.apply_chat_template(
                build_conversation(query, answer=None),
                add_generation_prompt=True,
            )

            full_enc = self.processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_enc = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = full_enc["input_ids"].clone()
            labels[0, :prompt_len] = -100        # mask prompt; predict answer only

            batch_input_ids.append(full_enc["input_ids"].squeeze(0))
            batch_attention_mask.append(full_enc["attention_mask"].squeeze(0))
            batch_pixel_values.append(full_enc["pixel_values"].squeeze(0))
            batch_labels.append(labels.squeeze(0))

        # Pad to longest sequence in the batch
        pad_id  = self.processor.tokenizer.pad_token_id
        max_len = max(t.shape[0] for t in batch_input_ids)

        def pad(t, val):
            return torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=val)

        return {
            "input_ids":      torch.stack([pad(t, pad_id) for t in batch_input_ids]),
            "attention_mask": torch.stack([pad(t, 0)      for t in batch_attention_mask]),
            "pixel_values":   torch.stack(batch_pixel_values),
            "labels":         torch.stack([pad(t, -100)   for t in batch_labels]),
        }


# ── Loss logger callback ───────────────────────────────────────────────────────

class LossLoggerCallback(TrainerCallback):
    """Appends loss to a CSV every logging_steps — easy to plot for the report."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, "training_loss.csv")
        self.records: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.records.append({
                "step":  state.global_step,
                "epoch": round(state.epoch or 0, 4),
                "loss":  round(logs["loss"], 6),
            })
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
                writer.writeheader()
                writer.writerows(self.records)


# ── Loss curve plot ───────────────────────────────────────────────────────────

def plot_loss_curve(csv_path: str, output_dir: str):
    """Read training_loss.csv and save a loss curve PNG."""
    steps, losses = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, linewidth=1.8, color="#2563EB", label="Training Loss")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("QLoRA Fine-tuning Loss Curve\n"
                 f"LLaVA-1.5-7B on ChartQA  |  {NUM_EPOCHS} epochs  |  {DATASET_SAMPLE_SIZE} samples",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "training_loss_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved → {out_path}")
    return out_path


# ── Main training entry point ─────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load model (4-bit) + processor
    model, processor = load_model_and_processor()

    # 2. Apply QLoRA (LoRA on LLM only; vision encoder stays frozen)
    model = apply_qlora(model)

    # 3. Load dataset (700 samples from ChartQA train split)
    dataset = load_train_dataset(DATASET_SAMPLE_SIZE)

    # 4. Data collator
    collator = LLaVAMultimodalCollator(processor, max_length=MAX_SEQ_LENGTH)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        remove_unused_columns=False,       # Must keep image/query/label columns
        report_to=["tensorboard"],
        dataloader_num_workers=0,          # Required on Windows
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[LossLoggerCallback(OUTPUT_DIR)],
    )

    # 7. Train!
    print("\nStarting QLoRA training…")
    trainer.train()

    # 8. Save LoRA adapter + processor
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"\nAdapter saved → {adapter_path}")

    # 9. Plot loss curve from CSV
    csv_path = os.path.join(OUTPUT_DIR, "training_loss.csv")
    if os.path.exists(csv_path):
        plot_loss_curve(csv_path, OUTPUT_DIR)

    # 10. Save training summary
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump({
            "model_id":       MODEL_ID,
            "dataset":        DATASET_NAME,
            "sample_size":    DATASET_SAMPLE_SIZE,
            "epochs":         NUM_EPOCHS,
            "lora_r":         LORA_R,
            "lora_alpha":     LORA_ALPHA,
            "target_modules": LORA_TARGET_MODULES,
            "adapter_path":   adapter_path,
        }, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
