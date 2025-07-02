# Sentiment Analysis on Tweets — Logistic Regression (From Scratch vs. Scikit-learn)

## 📌 Overview

This project performs **binary sentiment classification** on tweet data (positive vs. negative).  
It involves preprocessing text, extracting both textual and contextual features, and training a logistic regression classifier using two separate approaches:

-  **From Scratch Implementation** using NumPy
-  **Scikit-learn Implementation** with optimized hyperparameters

The goal is to evaluate both models, compare their performance, and understand what factors contribute to the best possible results.

---

## Dataset Summary

- Each instance includes:
  - `text`: the tweet content
  - `Platform`, `Time of Tweet`, `Year`: contextual metadata
  - `sentiment`: the label (positive/negative)
- **Neutral labels** were excluded to keep the task binary.

---

## Preprocessing

- Emojis converted to text (`emoji.demojize`)
- URL/user mentions/hashtags removed
- Lowercased and punctuation stripped
- Stopword removal using NLTK
- Lemmatization (`WordNetLemmatizer`)
- One-hot encoding for `Platform` and `Time of Tweet`
- TF-IDF on tweet text (bi-grams, max 5000 features)
- Standard scaling for `Year`

---

## Validation Setup

- **5-Fold Stratified Cross-Validation**
- Manual threshold tuning for each fold (`0.10` to `0.85` in steps of `0.05`)
- Metrics collected: **Accuracy, Precision, Recall, F1 Score**

---

## Implementation 1: Logistic Regression from Scratch

- Manual implementation of:
  - Sigmoid hypothesis
  - Gradient descent (with L2 regularization)
  - Bias term handling
- Trained for 200,000 iterations
- Outputs probabilities → tuned threshold → final predictions

### 
 Performance (Avg. Across Folds)
| Metric     | Value (%)     |
|------------|---------------|
| Accuracy   | 83.33%        |
| Precision  | 82.13%        |
| Recall     | 90.37%        |
| F1 Score   | 85.74%        |

## Drawback: Too slow, computationally expensive implementation. 
---

## Implementation 2: Scikit-learn Logistic Regression

- Used `LogisticRegression` from `sklearn.linear_model`
- Tuned hyperparameters:
  - `C=10000` (low regularization)
  - `solver='liblinear'` (suitable for small, sparse data)
  - `max_iter=200000` (to match custom convergence)
- Same preprocessing and evaluation pipeline

###  Performance (Avg. Across Folds)
| Metric     | Value (%)     |
|------------|---------------|
| Accuracy   | 83.67%        |
| Precision  | 83.62%        |
| Recall     | 88.56%        |
| F1 Score   | 85.73%        |

---

## **Comparative Analysis**

| Aspect                  | From Scratch                    | Scikit-learn                          |
|-------------------------|----------------------------------|---------------------------------------|
| Implementation          | Manual gradient descent         | Optimized built-in library            |
| Training Time           | Slower due to manual loops      | Faster using compiled C solvers       |
| Regularization Control  | Manual λ = 0.0001               | Equivalent via `C = 10000`            |
| Threshold Optimization  | Manual loop                     | Manual loop (same logic)              |
| Final F1 Score          | 85.74%                          | 85.73%                                |
| Maintainability         | Low                             | High                                  |
| Deployment Ready        | Needs conversion                | Directly usable with `.predict()`     |

Both models performed **equally well**. The small differences in precision/recall are natural due to numerical nuances. The final F1 scores are **nearly identical**, validating the correctness of both pipelines.

---

## Key Takeaways

- From-scratch implementation helped understand the mathematical intuition and gave full control over training.
- Scikit-learn version, when tuned properly, matches custom models and is easier to use, scale, and deploy.
- Threshold tuning is **crucial** for F1 optimization in imbalanced settings.

---
## Dataset

The dataset used in this project contains tweet data with metadata such as platform, time, year, and sentiment labels.

**Download**: [Sentiment Analysis Dataset (link)](https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis)

---
