# Multi-Layer Perceptrons Workshop

**Course:** CSCN 8010 — Foundations of Machine Learning Frameworks  
**Program:** Applied AI & Machine Learning — Conestoga College  
**Semester:** Winter 2025

---

## Team

| Name | GitHub | Role |
|---|---|---|
| Emmanuel | [@chooksemmanuel](https://github.com/chooksemmanuel) | Developer |
| Liggia Elena Taboada Cruz | [@liggiaelena](https://github.com/liggiaelena) | Developer / Repo Owner |
| Chao-Chung Liu (Thomas) | [@caatat741213](https://github.com/caatat741213) | Developer |

---

## Overview

This workshop extends our understanding of neural networks from single-layer perceptrons into full multi-layer architectures. The work is split into two challenges:

**Challenge 1 — Framework Comparison (Keras / PyTorch / TensorFlow)**  
We implement the same MLP architecture across all three major deep learning frameworks, train each one on the MNIST handwritten digit dataset, and compare training time, test accuracy, and developer experience.

**Challenge 2 — From-Scratch Implementation (NumPy only)**  
We build a 3-layer MLP (`Input(2) → Hidden-1(4, ReLU) → Hidden-2(4, ReLU) → Output(1, Sigmoid)`) using only NumPy — no Keras, no PyTorch, no TensorFlow. This means writing forward propagation, backpropagation, and gradient descent by hand. We also run a series of experiments comparing activation functions, learning rates, network depth, and visualising the forward activation flow and backward gradient flow.

---

## Repository Structure

```
MultiLayerPerceptrons_Workshop/
│
├── MLP_Workshop_Completed.ipynb        ← Main submission (Challenge 1 + 2, all answers)
├── MultiLayerPerceptrons_Workshop.ipynb ← Original professor template
├── MultiLayeredPerceptrons.ipynb        ← Additional working notebook
├── data/
│   └── MNIST/raw/                       ← MNIST dataset (auto-downloaded by PyTorch)
├── requirements.txt                     ← Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/liggiaelena/MultiLayerPerceptrons_Workshop.git
cd MultiLayerPerceptrons_Workshop
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook MLP_Workshop_Completed.ipynb
```

Or open in Google Colab / VS Code with the Jupyter extension.

> **Note:** MNIST data for PyTorch will be auto-downloaded to `./data/MNIST/raw/` on first run.

---

## Requirements

See `requirements.txt`. Key packages:

- `tensorflow` >= 2.x
- `torch` + `torchvision`
- `numpy`
- `matplotlib`

---

## Key Results

### Challenge 1 — Framework Comparison (MNIST, 5 epochs, batch size 64)

| Framework | Test Accuracy | Ease of Use | Best For |
|---|---|---|---|
| Keras | ~97–98% | ⭐⭐⭐⭐⭐ | Quick prototypes & learning |
| PyTorch | ~97–98% | ⭐⭐⭐⭐ | Research & custom architectures |
| TensorFlow Core | ~97–98% | ⭐⭐⭐ | Production & deployment at scale |

All three frameworks achieve similar accuracy. Keras is the fastest to write; PyTorch gives the most control; TensorFlow Core API is the most production-ready.

### Challenge 2 — From-Scratch MLP (NumPy, XOR-like toy dataset)

- Successfully trains a 3-layer MLP using hand-coded forward prop and backpropagation
- ReLU hidden layers converge significantly faster than Sigmoid due to avoiding the vanishing gradient problem
- Optimal learning rate on this task: `lr = 0.1` (balances speed and stability)
- Moderate depth (2–3 hidden layers) performs best; very deep networks (4+) suffer from vanishing gradients in earlier layers

---

## Concepts Covered

- Multi-layer neural network architecture
- Forward propagation: computing activations layer by layer
- Backpropagation: chain rule, gradient computation, parameter updates
- Activation functions: ReLU vs Sigmoid in hidden layers
- Binary cross-entropy loss
- Vanishing & exploding gradients
- Effect of learning rate and network depth on convergence
- Framework-level comparison: Keras, PyTorch, TensorFlow Core API
