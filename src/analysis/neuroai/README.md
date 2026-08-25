
# 🧠 Neuro-vs-AI 🤖 : Biological vs. Artificial Vision

This repository provides tools to compare humans and artificial neural networks (ANNs) across key cognitive and perceptual benchmarks, including:
- 🟪 Shape Bias
- 🟦 Degradation Robustness
- 🟥 Adversarial Robustness
- 🟨 (More coming soon…)

---
## ⚙️ Installation

To use the evaluation tools in this repo, clone it and install in editable mode:


    git clone https://github.com/KietzmannLab/Neuro-vs-AI.git
    cd Neuro-vs-AI
    pip install -e .

---

## 🎯 Project Goal

The purpose of this toolkit is to rigorously assess and compare the visual behavior of humans and ANNs, with a focus on divergences in robustness, generalization, and perceptual strategies. Our aim is to support cognitive benchmarks that go beyond standard accuracy metrics.

---

### 🔎 Shape Bias Evaluation

Shape bias refers to a system’s tendency to categorize objects based on shape rather than other visual features such as texture. This is a hallmark of human vision that many ANNs fail to replicate.

### 🔬 Usage Example: Evaluate Shape Bias for a Single Model

To evaluate shape bias for one model, run:

    
    python scripts/simple_evaluate.py
    


📤 Output:

The script prints the model’s shape bias score, e.g.:

shape bias (mean & median) in both behaviour & representation

A higher value means the model is more shape-biased.

---

### 📊 Compare Shape Bias Across Models

To compare shape bias across multiple models, run:

    python scripts/evaluate.py


This script loops over a list of model configurations and computes shape bias for each.

✍️ Example Configuration (in scripts/evaluate.py):


		models_to_compare = [
			{"model_name": "resnet50_TVD_B", "checkpoint": "...", "trained_on": "ecoset"},
			{"model_name": "vit_base_patch16", "checkpoint": None, "trained_on": "imagenet"},
			...
		]

Optionally, results can be saved or visualized for comparison.

---

## ✍️ Citation

If you use the code in your research, please consider to cite our work:

```bibtex
@article{lu2025dvd,
  title   = {Adopting a human developmental visual diet yields robust, shape-based AI vision},
  author  = {Lu, Zejin and Thorat, Sushrut and Cichy, Radoslaw M. and Kietzmann, Tim C.},
  journal = {arXiv preprint arXiv:2507.03168},
  year    = {2025},
  doi     = {10.48550/arXiv.2507.03168},
  url     = {https://arxiv.org/abs/2507.03168}
}
```
