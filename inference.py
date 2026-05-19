"""
inference.py – Inference with LLaVA-1.5-7B + QLoRA adapter.

Modes
-----
--mode zero_shot   Part 1: base model only (record original answers)
--mode compare     Part 3: side-by-side [Image | Base | Fine-tuned]  (default)

Usage
-----
  python inference.py --mode zero_shot
  python inference.py --mode compare
  python inference.py --mode compare --num_samples 5 --adapter_path ./llava-chartqa-finetuned/final_adapter
"""

import os
import json
import argparse
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from config import (
    MODEL_ID, ADAPTER_DIR, TEST_SAMPLE_SIZE,
    BNB_LOAD_IN_4BIT, BNB_QUANT_TYPE, BNB_DOUBLE_QUANT, BNB_COMPUTE_DTYPE,
    DATASET_NAME,
)
from load import load_test_dataset, get_answer, build_conversation

MAX_NEW_TOKENS = 80    # enough for base model verbose output


# ── Model helpers ─────────────────────────────────────────────────────────────

def create_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=BNB_LOAD_IN_4BIT,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
    )


def load_base_model():
    """Load base LLaVA in 4-bit quantisation."""
    print(f"Loading base model: {MODEL_ID}")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=create_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def attach_adapter(model, adapter_path: str):
    """Attach a trained LoRA adapter to an existing model."""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found at '{adapter_path}'.\n"
            "Run  python train.py  first."
        )
    print(f"Loading adapter: {adapter_path}")
    ft_model = PeftModel.from_pretrained(model, adapter_path)
    ft_model.eval()
    return ft_model


# ── Single-sample inference ───────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(model, processor, image, question: str,
                  short_answer: bool = False) -> str:
    """Generate an answer for one image-question pair.

    short_answer=True: return only the first whitespace-delimited token
                       (best for fine-tuned model which outputs correct
                       answer first then sometimes adds noise).
    """
    image = image.convert("RGB")
    prompt = processor.apply_chat_template(
        build_conversation(question, answer=None),
        add_generation_prompt=True,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    max_tok = 10 if short_answer else MAX_NEW_TOKENS
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tok,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True
    )[0].strip()
    if short_answer:
        return raw.split()[0] if raw else raw
    return raw


# ── Evaluation modes ──────────────────────────────────────────────────────────

def zero_shot_mode(num_samples: int):
    """Part 1 – run base model on test images and record answers."""
    print("\n" + "=" * 70)
    print("PART 1 – ZERO-SHOT BASELINE  (Base Model Only)")
    print("=" * 70)

    img_dir = "zero_shot_images"
    os.makedirs(img_dir, exist_ok=True)

    model, processor = load_base_model()
    test_data = load_test_dataset(num_samples)

    results = []
    for i, sample in enumerate(test_data):
        q   = sample["query"]
        gt  = get_answer(sample["label"])
        img = sample["image"].convert("RGB")
        ans = run_inference(model, processor, img, q)

        img_path = os.path.join(img_dir, f"sample_{i + 1}.png")
        img.save(img_path)

        print(f"\n[Sample {i + 1}/{num_samples}]")
        print(f"  Question     : {q}")
        print(f"  Ground Truth : {gt}")
        print(f"  Base Model   : {ans}")
        print(f"  Image saved  : {img_path}")

        results.append({
            "sample_id": i + 1,
            "question": q,
            "ground_truth": gt,
            "base_model_answer": ans,
            "image_path": img_path,
        })

    out_file = "zero_shot_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved  → {out_file}")
    print(f"Images saved   → {img_dir}/")
    return results


def compare_mode(num_samples: int, adapter_path: str):
    """Part 3 – side-by-side comparison: Base vs Fine-tuned."""
    print("\n" + "=" * 70)
    print("PART 3 – COMPARISON: Base Model vs Fine-tuned Model")
    print("=" * 70)

    # Load once; toggle adapter for base vs fine-tuned answers
    model, processor = load_base_model()
    ft_model = attach_adapter(model, adapter_path)
    test_data = load_test_dataset(num_samples)

    results = []
    for i, sample in enumerate(test_data):
        q  = sample["query"]
        gt = get_answer(sample["label"])
        img = sample["image"]

        # Base model answer – verbose output (disable LoRA adapter)
        with ft_model.disable_adapter():
            base_ans = run_inference(ft_model, processor, img, q, short_answer=False)

        # Fine-tuned answer – short_answer=True extracts first token
        # (model learned to start with correct answer; may not stop cleanly yet)
        ft_ans = run_inference(ft_model, processor, img, q, short_answer=True)

        improved = (
            ft_ans.lower().strip() == gt.lower().strip()
            and base_ans.lower().strip() != gt.lower().strip()
        )

        tag = "  [IMPROVED]" if improved else ""
        print(f"\n[Sample {i + 1}/{num_samples}]")
        print(f"  Question     : {q}")
        print(f"  Ground Truth : {gt}")
        print(f"  Base Model   : {base_ans}")
        print(f"  Fine-tuned   : {ft_ans}{tag}")

        results.append({
            "sample_id": i + 1,
            "question": q,
            "ground_truth": gt,
            "base_model_answer": base_ans,
            "finetuned_answer": ft_ans,
        })

    out_file = "comparison_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Pretty comparison table ──
    print("\n" + "=" * 100)
    print("COMPARISON TABLE  [Image | Base Model | Fine-tuned | Ground Truth]")
    print("=" * 100)
    hdr = f"{'#':<4}{'Question':<38}{'Base Model':<20}{'Fine-tuned':<20}{'Ground Truth':<18}"
    print(hdr)
    print("-" * 100)
    for r in results:
        print(
            f"{r['sample_id']:<4}"
            f"{r['question'][:36]:<38}"
            f"{r['base_model_answer'][:18]:<20}"
            f"{r['finetuned_answer'][:18]:<20}"
            f"{r['ground_truth'][:16]:<18}"
        )

    print(f"\nFull results saved → {out_file}")
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Inference Script")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "compare"],
        default="compare",
        help="zero_shot = Part 1 (base model); compare = Part 3 (base vs fine-tuned)",
    )
    parser.add_argument("--num_samples",   type=int, default=TEST_SAMPLE_SIZE)
    parser.add_argument("--adapter_path",  type=str, default=ADAPTER_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "zero_shot":
        zero_shot_mode(args.num_samples)
    else:
        compare_mode(args.num_samples, args.adapter_path)


if __name__ == "__main__":
    main()
