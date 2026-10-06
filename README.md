# APS Failure Cost Optimisation (Scania Trucks)

A cost-sensitive machine-learning pipeline that predicts failures of the Air Pressure System (APS) in heavy Scania trucks and chooses the decision threshold that minimises total maintenance cost. Instead of optimising accuracy, the model is built and evaluated against the real business cost of each type of mistake.

---

## Problem

Each truck record contains anonymised sensor readings. The goal is to flag trucks whose APS component is about to fail.

| Outcome | Meaning | Cost |
|---|---|---|
| False positive (FP) | A healthy truck is sent for an unnecessary check | **10** |
| False negative (FN) | A failing truck is missed | **500** |

The objective is to minimise **total cost = 10 · FP + 500 · FN**.

---

## Dataset

Fetch the dataset at https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

Download and place these two files in the project folder:

```
aps_failure_training_set.csv
aps_failure_test_set.csv
```

Each file has a `class` column (`pos` = APS failure, `neg` = other component failure) and 170 anonymised numeric sensor features. Missing values are encoded as `na`.

---

## Project structure

```
.
├── aps_cost_optimization.py        # Full pipeline
├── aps_failure_training_set.csv    # Training data (from UCI)
├── aps_failure_test_set.csv        # Official test data (from UCI)
├── results.csv                     # Generated: cost comparison table
├── cost_curve.png                  # Generated: cost vs threshold
├── feature_importance.png          # Generated: top 25 features
└── README.md
```

---

## Pipeline

### 1. Missing-value handling
- Missingness is computed on the **training set only**.
- Features with more than 70% missing values are dropped.
- No imputation is applied: the histogram-based gradient boosting model routes missing values to the best side of every split, so the fact that a value is missing is kept as a signal.

### 2. Class imbalance via cost-sensitive learning
- Failures are up-weighted in the loss with `pos_weight = (# negatives) / (# positives)`.
- No synthetic rows or resampling (no SMOTE) are used; the original data distribution is preserved.

### 3. Feature selection
- A model is trained on 75% of the training data and **permutation importance** (scored by average precision) is measured on the remaining 25%.
- The top 50% of features by importance are kept.

### 4. Threshold selection
- 5-fold stratified cross-validation produces **out-of-fold probabilities** on the training set.
- A sweep of thresholds from 0.01 to 0.99 is evaluated with the 10/500 cost function, and the threshold with the lowest cost is selected.

### 5. Final evaluation
- The final model is fit on the full training set and evaluated once on the official test set.
- Results are compared across four settings:
  - Selected-feature model at the default 0.5 threshold
  - All-feature model at the default 0.5 threshold
  - Selected-feature model at the cross-validated threshold
  - Reference: threshold chosen with knowledge of the test set (oracle)
- A naive baseline that never flags a truck is reported for context.

---

## Model

`HistGradientBoostingClassifier` (scikit-learn) with:

| Parameter | Value |
|---|---|
| `max_iter` | 300 |
| `learning_rate` | 0.05 |
| `max_leaf_nodes` | 31 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 1.0 |

An equivalent XGBoost configuration is `XGBClassifier(scale_pos_weight=pos_weight, tree_method="hist")`.

---

## Getting started

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Add the data
Download the two CSV files from the UCI link above into the project folder.

### 3. Run

```bash
python aps_cost_optimization.py
```

---

## Outputs

| File | Description |
|---|---|
| `results.csv` | Threshold, FP, FN, TP, TN, total cost and accuracy for each setting |
| `cost_curve.png` | Total cost against probability threshold on the test set, with the train out-of-fold curve, the chosen threshold and the 0.5 default (linear and log scale) |
| `feature_importance.png` | Top 25 features by permutation importance |

The console output also reports the share of rows containing missing values, the dropped features, the class weight, the kept features with the top 10 ranked, the chosen threshold and the full comparison table.

---

## Tech stack

- **Python**
- **pandas / NumPy** – data handling
- **scikit-learn** – gradient boosting, cross-validation, permutation importance, metrics
- **Matplotlib** – cost curve and feature-importance plots

---

## Key concepts

- **Cost-sensitive learning** with class weights in place of resampling
- **Native missing-value handling** in tree models
- **Permutation importance** for feature selection
- **Out-of-fold threshold tuning** against an asymmetric business cost
- **Leakage-free evaluation**: every choice (dropped features, selected features, threshold) is made on training data only, and the test set is scored once
