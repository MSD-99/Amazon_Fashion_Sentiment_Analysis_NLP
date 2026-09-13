# Amazon Fashion Reviews: NLP Sentiment Classification & Hyperparameter Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MSD-99/Amazon_Fashion_Sentiment_Analysis_NLP/blob/main/Amazon_Fashion_Sentiment_Analysis_NLP.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Bag%20of%20Words%20%7C%20TF--IDF-0052CC?style=for-the-badge&logo=scikitlearn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An NLP experiment on the **Amazon Fashion Reviews dataset** (UCSD Amazon Reviews 2023), with linear and tree-based classifiers, coefficient inspection, and grid-search hyperparameter tuning.

---

## 📌 Problem Formulation & Methodology

Analyzing customer feedback in e-commerce requires robust text representation and classification algorithms to identify polarity:

1. **Sentiment Label Formulation:**
   - $\text{Rating} \in \{1, 2\} \implies \textbf{NEGATIVE}$
   - $\text{Rating} = 3 \implies \textbf{NEUTRAL}$
   - $\text{Rating} \in \{4, 5\} \implies \textbf{POSITIVE}$
2. **Balanced Sampling:** Mitigating majority-class bias by targeting high-volume products with balanced positive and negative polarity distributions.
3. **Text Vectorization:** Constructing sparse feature spaces via tokenization, lowercasing, stopword removal, and $N$-gram Bag-of-Words / TF-IDF representations.
4. **Model Zoo Comparison:** Evaluating **Logistic Regression (LR)**, **Support Vector Machines (Linear SVM)**, and **Decision Trees (DT)**.
5. **Model Interpretability:** Extracting the strongest positive and negative linguistic coefficient drivers.
6. **Hyperparameter Tuning:** Cross-validated `GridSearchCV` over regularization strengths $C \in \{0.1, 1, 10, 100\}$ and optimization solvers.

---

## 🔬 Recorded Model Comparison

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Key Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Tuned)** | **82.00%** | **0.82** | **0.82** | **0.82** | $C=100$, `liblinear` in the recorded grid search. |
| **Linear SVM** | 80.00% | 0.80 | 0.80 | 0.80 | Linear kernel. |
| **Decision Tree Classifier** | 74.00% | 0.74 | 0.74 | 0.74 | Default estimator in the recorded run. |

These values come from a seeded 80/20 split of 250 balanced positive/negative reviews from one product, leaving only 50 test reviews. Neutral reviews are excluded. The notebook also fits `CountVectorizer` before the train/test split, which exposes the test vocabulary during feature construction. Treat the table as a small exploratory result, not a general Amazon Fashion benchmark.

---

## 📊 Confusion Matrices & Empirical Visualizations

<p align="center">
  <img src="figures/lr_confusion_matrix.png" width="32%" alt="Logistic Regression CM" />
  <img src="figures/svm_confusion_matrix.png" width="32%" alt="Linear SVM CM" />
  <img src="figures/dt_confusion_matrix.png" width="32%" alt="Decision Tree CM" />
</p>

---

## 🔍 Top Sentiment Indicator Keywords (Logistic Regression Coefficients)

| Rank | Top Positive Sentiment Keywords ($\beta > 0$) | Top Negative Sentiment Keywords ($\beta < 0$) |
| :---: | :--- | :--- |
| 1 | `great`, `love`, `perfect`, `comfortable` | `cheap`, `poor`, `disappointed`, `returned` |
| 2 | `fits`, `nice`, `good`, `quality` | `small`, `tight`, `thin`, `ripped` |
| 3 | `soft`, `happy`, `recommend`, `cute` | `uncomfortable`, `waste`, `bad`, `wrong` |

---

## 📁 Repository Structure

```text
├── Amazon_Fashion_Sentiment_Analysis_NLP.ipynb  # Interactive notebook with all execution logs & charts
├── data/
│   └── amazon_fashion_sample.jsonl             # 2,000-review convenience sample
├── figures/                                    # Exported high-resolution confusion matrix figures
│   ├── lr_confusion_matrix.png
│   ├── svm_confusion_matrix.png
│   └── dt_confusion_matrix.png
├── requirements.txt                            # Dependencies
├── .gitignore                                  # Git exclusions (ignores raw 2.4GB JSONL)
└── README.md                                   # Research documentation
```

---

## 🚀 Quickstart & Interactive Reproduction

### Option A: Google Colab (One-Click Instant Run)
Click to open and execute directly in Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MSD-99/Amazon_Fashion_Sentiment_Analysis_NLP/blob/main/Amazon_Fashion_Sentiment_Analysis_NLP.ipynb)

### Option B: Local Execution
```bash
# Clone the repository
git clone https://github.com/MSD-99/Amazon_Fashion_Sentiment_Analysis_NLP.git
cd Amazon_Fashion_Sentiment_Analysis_NLP

# Install dependencies
pip install -r requirements.txt

# Launch JupyterLab
jupyter lab Amazon_Fashion_Sentiment_Analysis_NLP.ipynb
```

The committed notebook expects the full `Amazon_Fashion.jsonl` under `src/data/amazon/`. To use the smaller committed sample, change `DATA_PATH`/`data_files` in the loading cell and verify that the selected product has enough reviews in both sentiment classes.

## License

Released under the [MIT License](LICENSE).
