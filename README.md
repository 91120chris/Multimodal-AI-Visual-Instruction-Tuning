### Part 1: Baseline Inference (Zero-Shot 測試)
```bash
python inference.py --mode zero_shot
```
### Part 2: QLoRA Fine-tuning (模型視覺指令微調)
```bash
python train.py
```
### Part 3: Evaluation & Comparison (微調前後結果對比)
```bash
python inference.py --mode compare
```
