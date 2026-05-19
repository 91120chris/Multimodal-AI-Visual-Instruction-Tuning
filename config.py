import torch

# ── Model & Dataset ───────────────────────────────────────────────────────────
MODEL_ID          = "llava-hf/llava-1.5-7b-hf"
DATASET_NAME      = "HuggingFaceM4/ChartQA"
DATASET_SAMPLE_SIZE = 1000         # 500-1000 as required
TEST_SAMPLE_SIZE    = 5            # Zero-shot / comparison test samples

# ── Output Paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR   = "./llava-chartqa-finetuned"
ADAPTER_DIR  = "./llava-chartqa-finetuned/final_adapter"

# ── Training Hyperparameters ──────────────────────────────────────────────────
NUM_EPOCHS              = 3
BATCH_SIZE              = 1        # RTX 4090: 24 GB VRAM
GRAD_ACCUM_STEPS        = 8        # Effective batch = 8
LEARNING_RATE           = 2e-4
WARMUP_STEPS            = 15      # ~3% of 1000-sample / 8-accum / 3-epoch run
MAX_SEQ_LENGTH          = 1024     # LLaVA-1.5 uses 576 image tokens + text
LOGGING_STEPS           = 10
SAVE_STEPS              = 200

# ── LoRA Configuration ────────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]   # LLM attention only

# ── BitsAndBytes 4-bit Quantization ──────────────────────────────────────────
BNB_LOAD_IN_4BIT  = True
BNB_QUANT_TYPE    = "nf4"
BNB_DOUBLE_QUANT  = True
BNB_COMPUTE_DTYPE = torch.float16
