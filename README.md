<h1 align="center">📐 Probabilistic Graphical Models — CSE756 (MSc @ BRAC University)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Bayesian%20Networks-PGM-412991?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HMM-Sequence%20Models-00b39e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/BRAC%20University-MSc%20CSE-012d35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Grade-A%2B-success?style=for-the-badge"/>
</p>

<p align="center">
  MSc coursework applying <strong>Probabilistic Graphical Models</strong> to
  Human Action Recognition — comparing Naive Bayes, Bayesian Networks, and HMMs
  against a deep learning baseline (EfficientNetB3).
</p>

---

## 📚 Course Info

- **Course:** CSE756 — Probabilistic Graphical Models
- **Program:** MSc in Computer Science & Engineering
- **University:** BRAC University
- **Grade:** A+ (97–100) → CGPA 4.00

---

## 🎯 Task: Human Action Recognition via PGMs

Apply classical probabilistic models to HAR and benchmark against modern deep learning.

---

## 🔬 Models Implemented

| Model | Type | Use Case |
|---|---|---|
| **Naive Bayes** | Probabilistic classifier | Feature independence assumption |
| **Bayesian Network** | Directed graphical model | Captures conditional dependencies |
| **Hidden Markov Model (HMM)** | Sequence model | Temporal action sequences |
| **EfficientNetB3** | Deep CNN (baseline) | Transfer learning comparison |

---

## 🏗️ Pipeline

```
HAR Dataset
      │
      ▼
Feature Extraction
(handcrafted features for PGMs / CNN features for baseline)
      │
      ▼
PGM Models            Deep Learning Baseline
Naive Bayes      vs   EfficientNetB3
Bayesian Network
HMM
      │
      ▼
Evaluation & Comparison
(Accuracy, F1, Confusion Matrix)
      │
      ▼
Analysis: When do PGMs compete with deep learning?
```

---

## 💡 Key Findings

- HMMs outperform Naive Bayes on temporal action sequences
- Bayesian Networks excel when training data is limited
- EfficientNetB3 achieves higher raw accuracy but PGMs offer better interpretability
- PGMs require far less compute and are explainable — valuable in low-data regimes

---

## 🚀 Getting Started

```bash
git clone https://github.com/Ramisa-16/Probabilistic-Graphical-Models.git
cd Probabilistic-Graphical-Models
pip install -r requirements.txt
jupyter notebook
```

---

## 🛠️ Tech Stack

**Python** · **pgmpy** · **hmmlearn** · **scikit-learn** · **TensorFlow/Keras** · **NumPy** · **Matplotlib**

---

## 👩‍💻 Author

**Ramisa Sharar Nidhi** — MSc CSE @ BRAC University (CGPA 4.00/4.00)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ramisashararnidhi-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/ramisashararnidhi)
[![GitHub](https://img.shields.io/badge/GitHub-Ramisa--16-181717?style=flat&logo=github)](https://github.com/Ramisa-16)
