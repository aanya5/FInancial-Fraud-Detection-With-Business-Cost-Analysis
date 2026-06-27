[README_fraud.md](https://github.com/user-attachments/files/29406247/README_fraud.md)
# Financial Fraud Detection with Business Cost Analysis

A machine learning pipeline for detecting financial transaction fraud on a highly imbalanced dataset, with threshold tuning driven by real-world business cost simulation.

---

## Project Overview

Financial fraud detection is a high-stakes classification problem where the cost of a missed fraud far exceeds the cost of a false alarm. This project builds a Random Forest classifier on a dataset of over 6 million financial transactions, applies class-weight balancing and log-transformation to handle skew and imbalance, and evaluates model performance using business-relevant metrics rather than accuracy alone.

---

## Dataset

- **Source:** PaySim synthetic financial transaction dataset (`Fraud.csv`)
- **Size:** 6,362,620 transactions
- **Fraud Rate:** ~0.13% (highly imbalanced)
- **Features:** Transaction type, amount, origin and destination account balances, timestamps

> The full dataset exceeds GitHub's file size limits and is not included. A sampled version is provided to allow notebooks to run during review.

---

## Methodology

### Data Cleaning
- Removed records with missing `isFraud` labels (target variable cannot be inferred)
- Zero balances for merchant accounts were retained per the data dictionary
- Applied `log1p` transformation to transaction amount to reduce right skew
- Dropped identifier columns (`nameOrig`, `nameDest`) to prevent overfitting
- Removed `isFlaggedFraud` to avoid data leakage

### Feature Engineering
- `balance_change_orig`: difference between origin account balance before and after transaction
- `balance_change_dest`: difference between destination account balance before and after transaction
- These features reduce multicollinearity between raw balance columns and directly capture anomalous fund movement

### Model
- **Algorithm:** Random Forest Classifier
- **Class weights:** `balanced` to compensate for severe class imbalance
- **Train/test split:** 70/30 with stratification

### Evaluation
- Precision, Recall, F1-score (fraud class)
- Confusion Matrix
- ROC-AUC Score: **0.9994**
- Precision-Recall AUC
- Threshold tuning across 0.1, 0.3, 0.5, 0.7 to balance recall vs. precision

### Business Cost Simulation
A cost function was applied to the confusion matrix:
- **False Negative (missed fraud):** ₹50,000
- **False Positive (wrongly flagged):** ₹500

This allows threshold selection to be driven by financial impact rather than statistical metrics alone.

---

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.9994 |
| Fraud Recall (default threshold) | ~98% |
| Fraud Precision (default threshold) | ~14% |
| Estimated Business Cost (threshold = 0.30) | ₹12,916,500 |

> Precision is intentionally lower; in fraud detection, missing a fraud is far more costly than reviewing a legitimate transaction.

---

## Top Predictors of Fraud

Based on feature importance from the Random Forest model:

1. **Transaction type** — TRANSFER and CASH-OUT are disproportionately associated with fraud
2. **Transaction amount** — large, sudden transfers are a strong signal
3. **Origin account balance depletion** — accounts emptied in a single transaction
4. **Destination account balance spike** — abnormal inflows to destination accounts
5. **Balance change features** — capturing the movement pattern before and after

These align with known fraud behavior: fraudsters typically transfer large sums, drain accounts, and cash out immediately.

---

## Business Recommendations

- Deploy real-time transaction scoring using ML models
- Apply risk-based limits instead of static thresholds
- Trigger additional authentication for high-risk transactions
- Monitor velocity patterns (rapid transfer → cash-out sequences)
- Combine ML predictions with rule-based checks for a hybrid system
- Retrain models periodically as fraud patterns evolve

---

## Measuring Effectiveness

- Reduction in total fraud losses over time
- Fraud detection rate vs. false positive rate
- Customer complaint volume (friction from false positives)
- Manual review workload
- Model stability monitoring (data drift, threshold drift)

A/B testing can be used to compare fraud outcomes with and without ML-based intervention.

---

## Tech Stack

Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
