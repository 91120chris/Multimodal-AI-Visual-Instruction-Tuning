"""
load.py – Dataset loading and formatting utilities for ChartQA VQA fine-tuning.
"""

from datasets import load_dataset
from config import DATASET_NAME, DATASET_SAMPLE_SIZE, TEST_SAMPLE_SIZE


def get_answer(label) -> str:
    """Normalise label field, which may be a list or a plain string."""
    if isinstance(label, list):
        return label[0].strip() if label else ""
    return str(label).strip()


def load_train_dataset(sample_size: int = DATASET_SAMPLE_SIZE):
    """Download and sample ChartQA training split."""
    print(f"Loading {DATASET_NAME} (train, {sample_size} samples)…")
    dataset = load_dataset(DATASET_NAME, split=f"train[:{sample_size}]")
    print(f"  Loaded {len(dataset)} samples")
    return dataset


def load_test_dataset(num_samples: int = TEST_SAMPLE_SIZE):
    """Load ChartQA test split for evaluation."""
    print(f"Loading {DATASET_NAME} (test, {num_samples} samples)…")
    dataset = load_dataset(DATASET_NAME, split=f"test[:{num_samples}]")
    print(f"  Loaded {len(dataset)} samples")
    return dataset


def build_conversation(query: str, answer: str | None = None) -> list[dict]:
    """
    Build a LLaVA-style conversation dict.
    If answer is None, returns a prompt-only conversation (for inference).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
    return messages
