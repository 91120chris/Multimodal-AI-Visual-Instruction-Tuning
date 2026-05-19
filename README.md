```bash
pip install -r requirements.txt
```
### Part 1: Baseline Inference
```bash
python inference.py --mode zero_shot
```
### Part 2: QLoRA Fine-tuning
```bash
python train.py
```
### Part 3: Evaluation & Comparison
```bash
python inference.py --mode compare
```
