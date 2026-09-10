# E-Commerce Sentiment Analysis: From Traditional ML to Transformers with XAI

An end-to-end Natural Language Processing (NLP) pipeline designed to classify Turkish e-commerce product reviews into positive and negative sentiments. This project demonstrates the evolution from traditional machine learning algorithms (TF-IDF + XGBoost) to state-of-the-art Deep Learning architectures (Turkish BERT), utilizing Explainable AI (XAI) techniques to interpret model decisions.

## Project Overview

Understanding customer sentiment is critical for e-commerce platforms. While traditional Bag-of-Words (BoW) approaches can capture explicit keywords, they often fail to understand semantic context and sarcasm. This project solves that problem by implementing a dual-pipeline approach:
1. **Baseline Model:** An XGBoost classifier trained on TF-IDF vectors.
2. **Advanced Model:** A fine-tuned BERT Transformer (`dbmdz/bert-base-turkish-cased`) leveraging PyTorch to capture deep contextual relationships.

To ensure transparency, model predictions are evaluated using **LIME** and **SHAP**, exposing the "black-box" nature of both tree-based and deep learning models.

> **Dataset:** The Turkish E-Commerce Reviews dataset used in this project can be found [here](https://www.kaggle.com/datasets/burhanbilenn/duygu-analizi-icin-urun-yorumlari).

## Key Features

* **Data Preprocessing Pipeline:** Handled highly imbalanced Turkish review data, including tokenization, stop-word removal, and label encoding.
* **XGBoost & LIME:** Built a baseline model and demonstrated its limitations (e.g., misinterpreting contextual negations) using LIME TextExplainer.
* **Transformer Fine-Tuning:** Developed a custom PyTorch `Dataset` and `DataLoader` to fine-tune a pre-trained BERT model on Google Colab with GPU acceleration.
* **SHAP TextExplainer:** Applied game-theoretic SHAP values to visualize BERT's bidirectional attention mechanism, proving its ability to successfully decode complex contextual negations that traditional models missed.

## Tech Stack

* **Machine Learning & NLP:** Hugging Face `transformers`, PyTorch, XGBoost, Scikit-learn, NLTK
* **Explainable AI (XAI):** SHAP, LIME
* **Data Processing:** Pandas, NumPy
* **Environment:** Google Colab (T4 GPU), VS Code

## Performance Comparison

| Model | Representation | Accuracy | Macro F1-Score | Contextual Understanding |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | TF-IDF (1-2 Ngrams) | Moderate | Moderate | Poor (Fails on sarcasm/negation) |
| **BERT** | Sub-word Tokenization | **98%** | **0.82** | **Excellent (Captures bi-directional context)** |

*Note: BERT achieved a 90% Precision score on the minority negative class, eliminating the False Positive problem observed in the baseline model.*

## Author
**İmge Su Çakmak**