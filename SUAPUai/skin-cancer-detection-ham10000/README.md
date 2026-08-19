

# 🧬 SKIN-XAI — Skin Lesion Intelligence & Explainable AI Research Lab

> **A research-grade AI platform for skin lesion analysis, trustworthy medical computer vision, explainable deep learning, uncertainty estimation, multimodal learning, and clinically oriented decision-support research using the HAM10000 dataset.**

![Image](https://images.openai.com/static-rsc-4/D-EfmsopyLW__4SdODM1E2UoZxc1IjjF-PkJheAHL-RyCRHxDVdyqeVnZtlFOa_Xd-nPZCJ_iZI5W0QqN8y92ga6-dFAtP-1prAA_bJ_sL0kQBfABsSQHLsgb2GFgXwY6NLxqNVqeZqHIkSj5yVMEYUk3iWapvPGzYFI6mb1ZZoUtdYLIpoC3ClHr0xyK46W?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/UVoLk0mtjQ-jau4w4t0wdnokli12K9kvim32imKmEkBKDs9VAZfjZ1hcmaaHZT-0rSROvO89q2y22jHhjHBIa7B0ukGYryleec4nAEwcPdmWzlqMYeG3nUuZWDjo85_DG3KyGu1jgi04lvJ_bY3rkZX57iSnEsGMg6UShcUCnVkRBJITmMq8o5C9VZSqI1Ju?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/GlUDYqC-OjJqFIf9I3AsiVE-M5dSRl4-7__kADPmFl5SGX41rgaX1oDBmnFfZc-Dk8Z1fFCfckTZmjvQIM8m9gjpoMU6nRIx54lb-fLpQMMgcrruTwGSmAgVmbv44HDb1NFiCKLkLYm6xcjHZx3rjeWyRfZMYotKXLpVDR0rdbbiwsJXjeWxwhWYht_k6Hym?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/ako1HOHHq4Je5ykeSKMVz8TlXhJecxW-o305UteQFPoSgqAWPryhs79txs-CPukkpkb0IF2EBhEuVy30XGIjmdsIULDGgGr3kstZDk_bEXpKc0xI9RYAD4H6hC48d5vwZ_YX4iOaVBNluhotCREtd6k54fAef1Y3z-kmmlez9no9ygIKNIlozcNEgXtzftoL?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/oLmekMQjLrVopeNktuMmfPghkmwZpOOYu8SPq8gyrLXB81QGE1-FRI0CWyW6njbUNjMTp31HX28PYEyBeXv_llK7cayacfowRnCyOdAqR1Vr6PAdl61zz6KWGS3mSA98uDxU-9THsrAUFYz5GqWQvjvvNsPtc5QzHCHc3DR99-Wjk1GQ8nTvNUhw41ODZtYf?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/OY1zV_VSj8d1oYlhMGmWj3px_Vxzonm2t_SUnIT1S2Kiku1uCM38XfEVjS4kUzEfaz_35Y3IwxGXomm0_Yx4cQCnsR8Qn-ZIlDE2mdMRrlXo9nyIzHvEHpOfn9vxN0akJwuXAEc3X8Da3o9e5oOp-9hs3JrTDe_xKIhHUhfSd27x377uw6MaNFjd-WYfGIe8?purpose=fullsize)

## 🔬 Research Vision

**SKIN-XAI** is an evolving research platform for investigating how modern Artificial Intelligence can analyze dermoscopic skin lesions while providing **reproducible, explainable, uncertainty-aware, and scientifically measurable predictions**.

The project started as an AI-course project inspired by an existing CNN-based HAM10000 classification workflow.

It is now being extended toward a broader research direction:

```text
HAM10000
   ↓
Dataset Intelligence
   ↓
Leakage-Safe Evaluation
   ↓
Deep Learning Benchmark
   ↓
Explainable AI
   ↓
Uncertainty Estimation
   ↓
Robustness Testing
   ↓
Multimodal Learning
   ↓
AI-Assisted Clinical Decision Support
```

The long-term objective is **not simply to maximize accuracy**.

> **The goal is to build an experimentally rigorous framework for understanding when, why, and how AI should be trusted for skin-lesion analysis.**

---

# 🏆 Current Project Status

### Current implemented foundation

* ✅ HAM10000 dataset pipeline
* ✅ Dataset audit
* ✅ Leakage-aware grouped splitting
* ✅ Patient/lesion-aware evaluation
* ✅ Reproducible configuration
* ✅ Baseline CNN
* ✅ ResNet18
* ✅ ResNet34
* ✅ ResNet50
* ✅ DenseNet121
* ✅ EfficientNet-B0
* ✅ Transfer learning
* ✅ Data augmentation
* ✅ Fine-tuning
* ✅ Class-imbalance strategies
* ✅ Macro-F1 based checkpoint selection
* ✅ Model comparison
* ✅ Confusion matrices
* ✅ Per-class evaluation
* ✅ Training history
* ✅ Automated evaluation reports
* ✅ Experiment configuration

### Historical result

An earlier version of the project achieved approximately **89% validation accuracy**.

However, this repository now prioritizes **leakage-safe evaluation and reproducibility rather than reporting a single accuracy number as the main scientific contribution**.

---

# 🧠 Why This Project Is Different

A conventional project asks:

> "Can a neural network classify skin lesions?"

This project asks deeper questions:

### Research Questions

1. Can the model generalize to unseen patients?
2. Is the model learning pathology or dataset artifacts?
3. How does performance change across lesion classes?
4. Which architectures are actually robust?
5. How confident is the model when it is wrong?
6. Can we explain why the model made a prediction?
7. How sensitive is the model to image perturbations?
8. Does performance remain stable across demographic and acquisition variations?
9. Can multimodal information improve prediction?
10. Can AI identify cases where it should **not** make a confident prediction?

---

# 📊 Current Architecture Benchmark

The current benchmark evaluates:

| Model           | Purpose                           |
| --------------- | --------------------------------- |
| Baseline CNN    | Classical baseline                |
| ResNet18        | Lightweight residual baseline     |
| ResNet34        | Deeper residual architecture      |
| ResNet50        | Deeper representation             |
| DenseNet121     | Feature reuse / transfer learning |
| EfficientNet-B0 | Efficient modern CNN              |

The benchmark is designed so that models are evaluated using the same canonical dataset split.

---

# 🔐 Leakage-Safe Dataset Pipeline

Medical imaging datasets can produce misleadingly high results when related images appear across training and test sets.

This project therefore uses grouped splitting based on available metadata such as:

```text
Patient ID
     +
Lesion ID
     ↓
Group-aware Split
     ↓
Train / Validation / Test
```

The generated:

```text
results/data_split.csv
```

is treated as the canonical split for experiments.

This makes future model comparisons substantially more reproducible.

---

# 🧪 Current Experiment Pipeline

```text
Raw HAM10000
      ↓
Metadata Audit
      ↓
Image Path Validation
      ↓
Duplicate / Group Analysis
      ↓
Class Distribution
      ↓
Grouped Split
      ↓
Data Augmentation
      ↓
Model Training
      ↓
Validation Macro-F1
      ↓
Best Checkpoint
      ↓
Independent Test Evaluation
      ↓
Metrics + Plots
      ↓
Research Report
```

---

# 🚀 Phase 1 — Dataset Intelligence

### Current

* ✅ Metadata loading
* ✅ Path resolution
* ✅ Dataset audit
* ✅ Class distribution
* ✅ Grouped splitting
* ✅ Split export

### Upgrade

* [ ] Automated duplicate detection
* [ ] Near-duplicate image detection
* [ ] Image quality scoring
* [ ] Blur detection
* [ ] Resolution analysis
* [ ] Artifact detection
* [ ] Metadata consistency checking
* [ ] Dataset bias analysis
* [ ] Acquisition-source analysis
* [ ] Automated dataset health report

### Future output

```text
DATASET HEALTH REPORT
────────────────────────────
Images                 : ...
Unique lesions         : ...
Unique patients        : ...
Duplicate candidates   : ...
Low-quality images     : ...
Class imbalance        : ...
Potential leakage      : ...
Dataset health score   : ...
```

---

# 🧬 Phase 2 — Stronger Model Benchmark

Current models will become the baseline research suite.

### Planned additions

* [ ] ConvNeXt
* [ ] Vision Transformer
* [ ] Swin Transformer
* [ ] EfficientNet variants
* [ ] MobileNet
* [ ] ResNeXt
* [ ] Modern hybrid CNN/Transformer models

The goal is not to collect models randomly.

Instead:

```text
CNN
 ↓
Modern CNN
 ↓
Transformer
 ↓
Hybrid Architecture
 ↓
Proposed Architecture
```

Each model should be evaluated under the same experimental protocol.

---

# 🔥 Phase 3 — Explainable AI

A high-performing medical model should also provide evidence for its prediction.

### Planned systems

* [ ] Grad-CAM
* [ ] Grad-CAM++
* [ ] Integrated Gradients
* [ ] Occlusion analysis
* [ ] Attention visualization
* [ ] Saliency maps
* [ ] Region-level analysis

Example:

```text
Input Lesion
     ↓
AI Prediction
     ↓
Explainability Engine
     ↓
Important Image Regions
     ↓
Visual Explanation
```

Example output:

```text
Prediction: Melanoma

Confidence: 0.91

Important Regions:
██████████████
████████░░████
██████████████
```

The objective is to investigate whether the model focuses on **medically meaningful lesion regions** rather than irrelevant artifacts.

---

# 🎯 Phase 4 — Uncertainty-Aware AI

This is one of the most important upgrades.

Instead of:

```text
Prediction = Melanoma
```

the system should eventually provide:

```text
Prediction       : Melanoma
Probability      : 0.87
Uncertainty      : Medium
Reliability      : Moderate
Abstention       : No
```

### Planned methods

* [ ] Confidence calibration
* [ ] Temperature scaling
* [ ] Expected Calibration Error
* [ ] Brier score
* [ ] Predictive entropy
* [ ] Monte Carlo dropout
* [ ] Deep ensembles
* [ ] Out-of-distribution detection
* [ ] Selective prediction
* [ ] AI abstention

### Research Question

> **Can a medical AI system learn when it should say "I don't know"?**

This can become a major research component.

---

# 🛡️ Phase 5 — Robustness & Safety Laboratory

Medical AI should not only work on clean test images.

The project will investigate model robustness under:

* [ ] Image noise
* [ ] Blur
* [ ] Brightness changes
* [ ] Contrast changes
* [ ] Compression
* [ ] Cropping
* [ ] Rotation
* [ ] Color variation
* [ ] Acquisition variation

Pipeline:

```text
Original Image
      ↓
Perturbation Engine
      ↓
AI Model
      ↓
Prediction Stability
      ↓
Robustness Score
```

---

# 🧠 Phase 6 — Bias & Fairness Analysis

A model can have high overall accuracy while performing poorly on specific subgroups.

Future work:

* [ ] Demographic subgroup analysis where metadata supports it
* [ ] Per-group sensitivity
* [ ] Per-group specificity
* [ ] Per-group F1
* [ ] Calibration comparison
* [ ] Error distribution analysis
* [ ] Dataset bias investigation

Goal:

> **Measure where the model works, where it fails, and whether failures are systematically distributed.**

---

# 🔬 Phase 7 — Multimodal Skin Intelligence

A future research direction is moving beyond image-only classification.

```text
Dermoscopic Image
        +
Metadata
        +
Clinical Information
        ↓
Multimodal Encoder
        ↓
Fusion Layer
        ↓
Prediction
        +
Explanation
        +
Uncertainty
```

Potential inputs:

* lesion image
* patient metadata
* lesion location
* age
* sex
* lesion history
* acquisition information

This should only be implemented when appropriate data and research methodology are available.

---

# 🤖 Phase 8 — Vision-Language Medical AI

A longer-term research direction is combining visual models with language models.

Potential system:

```text
Skin Image
    ↓
Vision Encoder
    ↓
Lesion Representation
    ↓
Medical Reasoning Layer
    ↓
Language Model
    ↓
Structured Explanation
```

Potential output:

```text
Predicted Class:
Melanocytic Nevus

Visual Evidence:
...

Confidence:
...

Uncertainty:
...

Alternative Classes:
...

Reasoning Summary:
...

Recommendation:
Requires professional clinical evaluation.
```

### Important

This project is intended as a **research/decision-support system**, not a replacement for dermatologists or clinical diagnosis.

---

# 🧬 Phase 9 — Lesion Representation Learning

Instead of directly predicting seven classes, future models may learn a general representation of skin lesions.

```text
Image
 ↓
Vision Encoder
 ↓
Latent Lesion Representation
 ↓
 ┌───────────────┬───────────────┐
 ↓               ↓               ↓
Classification   Retrieval     Similarity
```

Potential research:

* [ ] Contrastive learning
* [ ] Self-supervised learning
* [ ] Metric learning
* [ ] Image embeddings
* [ ] Similar-lesion retrieval
* [ ] Few-shot classification

---

# 🔭 Phase 10 — Generative AI Research

Potential future experiments:

* [ ] Synthetic lesion generation
* [ ] Data augmentation using generative models
* [ ] Synthetic minority-class generation
* [ ] Image-to-image translation
* [ ] Generative robustness testing

Synthetic data should always be evaluated carefully for:

* realism
* diversity
* memorization
* distribution shift
* clinical validity

---

# 🧪 Phase 11 — Research Benchmark Engine

The project will eventually automate the complete experiment lifecycle.

```text
Experiment Config
       ↓
Dataset Version
       ↓
Model
       ↓
Training
       ↓
Evaluation
       ↓
Explainability
       ↓
Calibration
       ↓
Robustness
       ↓
Statistical Analysis
       ↓
Research Report
```

Example:

```yaml
experiment:
  name: densenet121_baseline_v3

model:
  name: densenet121

dataset:
  version: ham10000_v1

evaluation:
  grouped_split: true
  metrics:
    - accuracy
    - macro_f1
    - balanced_accuracy
    - roc_auc
```

---

# 📈 Phase 12 — Beyond Accuracy

The primary evaluation should become:

```text
Accuracy
Macro-F1
Balanced Accuracy
Precision
Recall
Specificity
Sensitivity
ROC-AUC
PR-AUC
Calibration
ECE
Brier Score
Robustness
Uncertainty
```

For each class:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

the system should generate independent metrics.

---

# 🧮 Statistical Validation

Future experiments should include:

* [ ] Confidence intervals
* [ ] Bootstrap evaluation
* [ ] Repeated experiments
* [ ] Seed sensitivity
* [ ] Statistical significance testing
* [ ] Model comparison tests

This helps answer:

> "Is Model A actually better than Model B, or did it simply get lucky on one split?"

---

# 🧪 Ablation Studies

Research-grade development requires understanding which components actually matter.

Example:

```text
Baseline
   ↓
+ Augmentation
   ↓
+ Transfer Learning
   ↓
+ Class Weighting
   ↓
+ Fine Tuning
   ↓
+ Calibration
   ↓
+ Proposed Component
```

Each modification should be measured independently.

---

# 🏗️ Target Research Architecture

```text
                    SKIN-XAI
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     DATASET         VISION          AI
   INTELLIGENCE      ENGINE        REASONING
        │              │              │
        └──────────────┼──────────────┘
                       ↓
               LESION REPRESENTATION
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        CLASSIFICATION     RETRIEVAL
              │                 │
              └────────┬────────┘
                       ↓
                UNCERTAINTY
                       ↓
                EXPLAINABILITY
                       ↓
                 ROBUSTNESS
                       ↓
               RESEARCH REPORT
```

---

# 📂 Target Repository Structure

```text
Skin_cancer_detection_ham/
│
├── configs/
│   └── default.yaml
│
├── data/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validate_experiment.py
│   └── run_experiment.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   ├── uncertainty/
│   ├── robustness/
│   ├── fairness/
│   └── utils.py
│
├── research/
│   ├── experiments/
│   ├── ablations/
│   ├── benchmarks/
│   └── reports/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── reports/
│
├── checkpoints/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🗺️ Development Roadmap

## Phase 1 — Foundation

* [x] HAM10000 pipeline
* [x] Dataset audit
* [x] Grouped split
* [x] Model benchmark
* [x] Reproducible configuration

## Phase 2 — Explainability

* [ ] Grad-CAM
* [ ] Saliency
* [ ] Attention visualization
* [ ] Explanation reports

## Phase 3 — Trustworthy AI

* [ ] Calibration
* [ ] Uncertainty
* [ ] Abstention
* [ ] OOD detection

## Phase 4 — Robust AI

* [ ] Perturbation testing
* [ ] Robustness benchmark
* [ ] Distribution-shift analysis

## Phase 5 — Fairness

* [ ] Subgroup analysis
* [ ] Bias detection
* [ ] Error analysis

## Phase 6 — Advanced AI

* [ ] Vision Transformers
* [ ] Self-supervised learning
* [ ] Contrastive learning
* [ ] Lesion embeddings

## Phase 7 — Multimodal AI

* [ ] Image + metadata
* [ ] Multimodal fusion
* [ ] Vision-language experiments

## Phase 8 — Research

* [ ] Ablation studies
* [ ] Statistical validation
* [ ] Benchmark suite
* [ ] Reproducibility package
* [ ] Technical report
* [ ] Research paper

---

# 🔗 Connection to My Broader AI Research

This project is one component of my broader research portfolio in:

```text
Artificial Intelligence
        │
        ├── Computer Vision
        ├── Medical AI
        ├── Deep Learning
        ├── Explainable AI
        ├── AGI Research
        ├── Biomedical Intelligence
        └── Intelligent Systems
```

It complements my other work in:

* **CardioAGI-X** — biomedical intelligence
* **HAM10000 Skin Cancer Detection** — medical computer vision
* **Sperm Detection / Male Infertility Analysis** — biomedical image analysis
* **SAGI** — broader intelligent-agent research
* **Space Intelligence** — autonomous intelligent systems

The objective is to build individual research systems first and progressively investigate how their underlying AI methodologies can contribute to larger intelligent systems.

---

# 🧑‍🔬 Research Principles

This project follows:

### Reproducibility

Every experiment should have:

* configuration
* dataset version
* random seed
* model version
* training parameters
* evaluation protocol

### Scientific Honesty

Unimplemented features are marked as **planned**.

Reported results should come from reproducible experiments.

### Model Transparency

A model should ideally answer:

> **What did I predict?**

> **Why did I predict it?**

> **How confident am I?**

> **When should I not trust my prediction?**

---

# ⚠️ Medical Disclaimer

This project is an **academic and research prototype**.

It is **not a medical device**, diagnostic system, or substitute for professional medical evaluation.

Predictions generated by this repository should not be used as the sole basis for medical decisions.

---

# 📚 Dataset

The project uses the **HAM10000 dataset**, containing seven major dermoscopic lesion categories:

| Code    | Lesion                                        |
| ------- | --------------------------------------------- |
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc`   | Basal cell carcinoma                          |
| `bkl`   | Benign keratosis-like lesions                 |
| `df`    | Dermatofibroma                                |
| `mel`   | Melanoma                                      |
| `nv`    | Melanocytic nevi                              |
| `vasc`  | Vascular lesions                              |

Dataset access, licensing, and research usage should follow the original dataset provider's terms.

---

# 🛠️ Technology Stack

### Deep Learning

* Python
* PyTorch
* TorchVision
* TIMM

### Computer Vision

* OpenCV
* Albumentations

### Machine Learning

* Scikit-learn
* NumPy
* Pandas

### Research

* YAML configuration
* Experiment tracking
* Statistical evaluation
* Visualization
* Reproducible pipelines

### Future

* Vision Transformers
* Explainable AI
* Uncertainty estimation
* Self-supervised learning
* Multimodal learning
* Vision-Language Models

---

# 🚀 Quick Start

Clone the repository:

```bash
git clone https://github.com/mdyusuf166/Skin_cancer_detection_ham.git
cd Skin_cancer_detection_ham
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the dataset:

```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```

Train:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/default.yaml
```

---

# 📊 Current Research Pipeline

```text
HAM10000
   ↓
Dataset Audit
   ↓
Leakage-Safe Split
   ↓
Baseline CNN
   ↓
ResNet
   ↓
DenseNet
   ↓
EfficientNet
   ↓
Macro-F1 Benchmark
   ↓
Explainability
   ↓
Uncertainty
   ↓
Robustness
   ↓
Multimodal Intelligence
```

---

# 🏆 Long-Term Goal

The long-term objective of **SKIN-XAI** is to investigate a new generation of medical AI systems that are:

> **Accurate + Explainable + Uncertainty-Aware + Robust + Reproducible**

rather than optimizing for accuracy alone.

The ultimate research question is:

> ### **Can we build a skin-lesion AI system that not only predicts correctly, but can explain its evidence, quantify its uncertainty, recognize its limitations, and remain reliable under real-world variation?**

---

# 👨‍💻 Author

**MD Yousuf**

Computer Science & Engineering Student
Research interests:

```text
AI
Machine Learning
Computer Vision
Biomedical AI
AGI
Neural Systems
Intelligent Systems
```

---

# ⭐ Project Evolution

```text
AI Course Project
       ↓
HAM10000 Classifier
       ↓
Multi-Model Benchmark
       ↓
Leakage-Safe Research Pipeline
       ↓
Explainable AI
       ↓
Trustworthy Medical AI
       ↓
Multimodal Intelligence
       ↓
Research Benchmark
       ↓
Publication-Ready Research
```

> **From a skin cancer classification project → toward a reproducible medical AI research laboratory.**
