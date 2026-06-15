<!-- ─────────────────────────────────────────────────────────────────────── -->
<!--                       TRUTHSIFT — README                              -->
<!-- ─────────────────────────────────────────────────────────────────────── -->

<div align="center">

```
████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗███████╗██╗███████╗████████╗
╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║██╔════╝██║██╔════╝╚══██╔══╝
   ██║   ██████╔╝██║   ██║   ██║   ███████║███████╗██║█████╗     ██║   
   ██║   ██╔══██╗██║   ██║   ██║   ██╔══██║╚════██║██║██╔══╝     ██║   
   ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║███████║██║██║        ██║   
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝  
```

### *Cutting Through the Noise — One Headline at a Time*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> **"In a world flooded with misinformation, we built a sieve."**  
> TruthSift is an end-to-end fake news detection system that benchmarks six classical and deep learning models, then introduces an optimized Bidirectional LSTM — *TruthSift* — that achieves **97.85% accuracy** on the WELFake dataset.

</div>

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Models Implemented](#-models-implemented)
- [TruthSift — The Star Model](#-truthsift--the-star-model)
- [Results at a Glance](#-results-at-a-glance)
- [Visualizations](#-visualizations)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Key Design Decisions](#-key-design-decisions)

---

## 🎯 The Problem

Fake news spreads faster than corrections. Traditional detection methods (manual fact-checking, rule-based filters) cannot scale to the billions of news articles published daily.

This project answers a direct question:

> *Can a machine learn to distinguish real journalism from fabricated content — reliably, efficiently, and at scale?*

We built a systematic comparison across six model families, from classic statistical approaches to deep sequential architectures, and then engineered a stronger solution on top of what we learned.

---

## 📦 Dataset

**WELFake Dataset**  
A widely-used benchmark for fake news classification, merging four existing news datasets into a single, balanced corpus.

| Property | Value |
|---|---|
| Total Samples | **60,491 articles** |
| Real News (Label 0) | 34,030 |
| Fake News (Label 1) | 26,461 |
| Features Used | `title` + `text` (concatenated) |
| Train / Validation Split | 90% / 10% (stratified) |
| Source | `WELFake_Dataset.csv` |

**Preprocessing Pipeline:**
```
Raw Text → Lowercasing → Punctuation Removal → Tokenization
       → Stop-word Removal → POS-aware Lemmatization → Cleaned Corpus
```

Word clouds for each class were generated to visually inspect vocabulary patterns before modeling.

---

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRUTHSIFT PIPELINE                       │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  Raw CSV │──▶│ Preprocessing│──▶│  Feature Engineering │    │
│  └──────────┘   └──────────────┘   └──────────┬───────────┘    │
│                                                │                 │
│               ┌────────────────────────────────┤                 │
│               │                                │                 │
│               ▼                                ▼                 │
│     ┌──────────────────┐            ┌─────────────────────┐     │
│     │  TF-IDF Vectors  │            │  Word2Vec / GloVe   │     │
│     │  (Classical ML)  │            │  Embeddings (DL)    │     │
│     └────────┬─────────┘            └──────────┬──────────┘     │
│              │                                 │                 │
│    ┌─────────▼──────────────────────────┐      │                 │
│    │  RF │ SVM │ SimpleNN │ RNN │ LSTM │      │                 │
│    └─────────────────────────────────────┘      │                 │
│                                                 │                 │
│              ┌──────────────────────────────────▼──────────┐    │
│              │         TruthSift (Optimized Bi-LSTM)        │    │
│              │   GloVe Init → Spatial Dropout → BiLSTM ×2  │    │
│              │   → Layer Norm → Sigmoid Output              │    │
│              └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented

Six baseline models were trained and evaluated before building the final model:

### Classical ML

| Model | Vectorization | Notes |
|---|---|---|
| **Random Forest (RF)** | TF-IDF | 100 trees, StratifiedKFold CV |
| **SVM** | TF-IDF | RBF kernel, GridSearchCV tuning |

### Neural Networks

| Model | Embedding | Architecture |
|---|---|---|
| **SimpleNN** | Keras Tokenizer | Embedding → Flatten → Dense |
| **RNN** | Word2Vec (100d) | SimpleRNN → Dense |
| **LSTM** | Word2Vec (100d) | LSTM → Dropout → Dense |
| **CNN** | Word2Vec (100d) | Conv1D → GlobalMaxPool → Dense |

All deep models used early stopping (`val_loss`, patience=3) and Adam optimizer.

---

## ⭐ TruthSift — The Star Model

After benchmarking all six baselines, a gap analysis was performed. The final model — dubbed **TruthSift** — is an optimized Bidirectional LSTM with targeted fixes for every observed underperformance factor.

### Architecture

```
Input Sequence (max_len=300)
        │
        ▼
┌──────────────────────────────────┐
│  Embedding Layer                 │
│  • GloVe 100d initialization     │
│  • trainable=True  ← KEY CHANGE  │
│  • L2 regularization (0.01)      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  SpatialDropout1D (0.3)          │
│  Drops entire feature maps       │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Bidirectional LSTM (128 units)  │
│  dropout=0.3, recurrent=0.1      │
│  return_sequences=True           │
└──────────────┬───────────────────┘
               │
               ▼
     Layer Normalization (ε=1e-6)
               │
               ▼
┌──────────────────────────────────┐
│  Bidirectional LSTM (64 units)   │
│  dropout=0.3, recurrent=0.1      │
│  return_sequences=False          │
└──────────────┬───────────────────┘
               │
               ▼
     Layer Normalization (ε=1e-6)
               │
               ▼
         Dropout (0.3)
               │
               ▼
     Dense(1, activation=sigmoid)
     + L2 regularization (0.001)
```

### Key Optimizations vs. Baseline LSTM

| Component | Baseline LSTM | TruthSift Bi-LSTM |
|---|---|---|
| Direction | Unidirectional | **Bidirectional** |
| Embedding | Frozen | **Trainable** |
| Embedding dim | 100 | **128** |
| Recurrent dropout | None | **0.1** |
| Layer Normalization | None | **After each LSTM** |
| LR Schedule | Fixed | **Exponential Decay** |
| Gradient Clipping | None | **clipnorm=1.0** |
| Class Weights | None | **Computed (imbalance)** |

**Optimizer:** Adam with `ExponentialDecay` (initial lr=1e-3, decay=0.96 every 500 steps)  
**Batch size:** 128 | **Max epochs:** 10 | **Early stopping patience:** 3

---

## 📊 Results at a Glance

### Full Model Comparison (Validation Set)

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 | **TruthSift (Bi-LSTM)** | **0.9785** | **0.9850** | 0.9656 | **0.9752** |
| 🥈 | LSTM | 0.9779 | 0.9805 | 0.9686 | 0.9745 |
| 🥉 | CNN | 0.9775 | 0.9776 | **0.9709** | 0.9742 |
| 4 | SVM | 0.9698 | 0.9642 | 0.9668 | 0.9655 |
| 5 | SimpleNN | 0.9646 | 0.9645 | 0.9543 | 0.9594 |
| 6 | RF | 0.9494 | 0.9415 | 0.9430 | 0.9422 |
| 7 | RNN | 0.9428 | 0.9492 | 0.9184 | 0.9336 |

### AUC / Average Precision (Baseline Reference)

| Model | AUC-ROC | Avg Precision |
|---|---|---|
| TruthSift | — (computed live) | — (computed live) |
| LSTM | 0.9919 | 0.9863 |
| CNN | 0.9913 | 0.9855 |
| SVM | 0.9856 | 0.9753 |
| SimpleNN | 0.9799 | 0.9684 |
| RF | 0.9672 | 0.9481 |
| RNN | 0.9614 | 0.9425 |

> TruthSift improved F1-Score over the average baseline by a measurable margin while achieving the highest precision (0.9850) — meaning fewer real news articles were falsely flagged as fake.

---

## 📈 Visualizations

The notebook generates the following plots automatically:

```
📊 training_curves/
   ├── optimized_bilstm_training_curves.png   ← Loss & accuracy per epoch
   ├── final_bilstm_vs_paper_baselines.png    ← Bar chart across 4 metrics
   ├── final_bilstm_roc_pr_curves.png         ← ROC + Precision-Recall
   ├── final_bilstm_detailed_eval.png         ← Confusion matrix + per-class bars
   └── final_bilstm_head_to_head.png          ← F1 & Acc vs top 3 baselines
```

Word clouds for Fake vs Real news vocabularies are also generated during EDA.

A **live interactive dashboard** (`dashboard.py`) is included, powered by Dash/Plotly, accessible at `http://127.0.0.1:8050` after launching.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/truthsift-fake-news.git
cd truthsift-fake-news
```

### 2. Install Dependencies


```
# Core requirements
numpy pandas matplotlib seaborn scipy tqdm wordcloud
nltk scikit-learn tensorflow gensim joblib
```

### 3. Download Required Files

| File | Source |
|---|---|
| `WELFake_Dataset.csv` | [Kaggle — WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| `glove.6B.100d.txt` | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

Place both files in the project root before running.

### 4. Run the Notebook

```bash
jupyter notebook Final_Project_Notebook.ipynb
```

Run cells in order. The notebook is self-contained — all model classes are embedded.

### 5. Launch the Dashboard (Optional)

```bash
cd fake-news/
python dashboard.py
# Open: http://127.0.0.1:8050
```

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow / Keras |
| **Classical ML** | Scikit-Learn |
| **Embeddings** | GloVe (Stanford), Gensim Word2Vec |
| **NLP Preprocessing** | NLTK (tokenize, lemmatize, POS-tag) |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Dashboard** | Dash / Plotly |
| **Training Environment** | Kaggle (dual NVIDIA T4 GPUs) |

---

## 📁 Project Structure

```
truthsift-fake-news/
│
├── Final_Project_Notebook.ipynb   ← Main notebook (all steps)
├── dashboard.py                   ← Interactive comparison dashboard
├── dashboard_data.pkl             ← Serialized metrics for dashboard
│
├── WELFake_Dataset.csv            ← Raw dataset (download separately)
├── WELFake_preprocessed.csv       ← Auto-generated after preprocessing
│
├── glove.6B.100d.txt              ← GloVe embeddings (download separately)
│
├── our_bilstm_model.keras         ← Saved TruthSift model weights
│
├── plots/
│   ├── optimized_bilstm_training_curves.png
│   ├── final_bilstm_vs_paper_baselines_cell25_style.png
│   ├── final_bilstm_roc_pr_curves.png
│   ├── final_bilstm_detailed_eval.png
│   └── final_bilstm_head_to_head.png
│
└── README.md
```

---
<div align="center">

---

*Built with intent. Benchmarked with rigor. Named with purpose.*  
**TruthSift** — because the truth deserves better tools.

</div>
