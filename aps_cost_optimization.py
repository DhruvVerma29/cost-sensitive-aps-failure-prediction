
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, accuracy_score


FP_COST, FN_COST = 10, 500
SEED = 42


def cost(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return FP_COST * fp + FN_COST * fn, fp, fn, tp, tn

def build_model(pos_weight):
    # Cost-sensitive learning: up-weight failures in the loss. No synthetic rows, no SMOTE.
    # XGBoost equivalent: XGBClassifier(scale_pos_weight=pos_weight, tree_method="hist")
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED)

def fit(model, X, y, pos_weight):
    w = np.where(y == 1, pos_weight, 1.0)
    return model.fit(X, y, sample_weight=w)

# ---------------- Phase 1: missingness ----------------
train = pd.read_csv("aps_failure_training_set.csv", skiprows=20, na_values="na")
test = pd.read_csv("aps_failure_test_set.csv", skiprows=20, na_values="na")

ytr = (train.pop("class") == "pos").astype(int).values
yte = (test.pop("class") == "pos").astype(int).values
Xtr = train.astype(float)
Xte = test.astype(float)
print(f"Train {Xtr.shape}, failures {ytr.mean():.2%} | Test {Xte.shape}, failures {yte.mean():.2%}")

miss = Xtr.isna().mean().sort_values(ascending=False)   # computed on TRAIN only
drop_cols = miss[miss > 0.70].index.tolist()
print(f"Features >70% missing (dropped): {drop_cols}")
print(f"Rows with >=1 NaN: {Xtr.isna().any(axis=1).mean():.1%}  (dropna() would destroy this much data)")
keep = [c for c in Xtr.columns if c not in drop_cols]
Xtr, Xte = Xtr[keep], Xte[keep]
# No imputation: tree learner routes NaN to the best side of each split (informative missingness retained).

# ---------------- Phase 2: imbalance via weights ----------------
pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
print(f"pos_weight (neg/pos) = {pos_weight:.1f}")

# ---------------- Phase 3: feature selection ----------------
# Importance measured on a held-out split (permutation importance, scored by recall of failures proxy: average precision)
Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=SEED)
m0 = fit(build_model(pos_weight), Xa, ya, pos_weight)
pi = permutation_importance(m0, Xb, yb, scoring="average_precision", n_repeats=3,
                            random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xtr.columns).sort_values(ascending=False)
top = imp.index[: len(imp) // 2].tolist()               # keep top 50%
print(f"Kept {len(top)}/{len(imp)} features. Top 10:\n{imp.head(10).round(4)}")
Xtr_s, Xte_s = Xtr[top], Xte[top]

# ---------------- Phase 4: threshold from out-of-fold probabilities ----------------
def oof_probs(X, y):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = fit(build_model(pos_weight), X.iloc[tr], y[tr], pos_weight)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof

oof_full = oof_probs(Xtr, ytr)
oof_sel = oof_probs(Xtr_s, ytr)

thrs = np.round(np.arange(0.01, 1.00, 0.01), 2)
cv_curve = np.array([cost(ytr, oof_sel, t)[0] for t in thrs])
best_thr = thrs[cv_curve.argmin()]
print(f"\nThreshold chosen on TRAIN out-of-fold predictions: {best_thr}")

# ---------------- Final model, evaluated once on the official test set ----------------
final = fit(build_model(pos_weight), Xtr_s, ytr, pos_weight)
p_te = final.predict_proba(Xte_s)[:, 1]
final_full = fit(build_model(pos_weight), Xtr, ytr, pos_weight)
p_te_full = final_full.predict_proba(Xte)[:, 1]

test_curve = np.array([cost(yte, p_te, t)[0] for t in thrs])
oracle_thr = thrs[test_curve.argmin()]

rows = []
def add(name, y, p, t):
    c, fp, fn, tp, tn = cost(y, p, t)
    rows.append(dict(setting=name, threshold=t, FP=fp, FN=fn, TP=tp, TN=tn, total_cost=c,
                     accuracy=round(accuracy_score(y, (p >= t).astype(int)), 4)))
add("Selected-feature model, default 0.5", yte, p_te, 0.5)
add("All-feature model, default 0.5", yte, p_te_full, 0.5)
add(f"Selected-feature model, CV-chosen thr", yte, p_te, best_thr)
add(f"[reference] test-oracle thr (not deployable)", yte, p_te, oracle_thr)
res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("results.csv", index=False)

# Naive baseline: never flag anything
print(f"\nBaseline 'never flag a truck': cost = {FN_COST * yte.sum()}  (accuracy {1 - yte.mean():.4f})")

# ---------------- Cost curve ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(thrs, test_curve, lw=2, color="#1f6feb", label="Test set")
ax[0].plot(thrs, cv_curve * len(yte) / len(ytr), lw=1.5, ls="--", color="#888", label="Train OOF (rescaled)")
ax[0].axvline(best_thr, color="#d1242f", ls=":", label=f"CV-chosen threshold = {best_thr}")
ax[0].axvline(0.5, color="k", ls=":", alpha=.5, label="Default 0.5")
ax[0].set(xlabel="Probability threshold", ylabel="Total cost (10·FP + 500·FN)", title="Cost curve")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(thrs, test_curve, lw=2, color="#1f6feb"); ax[1].set_yscale("log")
ax[1].set(xlabel="Probability threshold", ylabel="Total cost (log)", title="Same curve, log scale")
ax[1].grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig("cost_curve.png", dpi=150)
imp.head(25)[::-1].plot.barh(figsize=(6, 6), title="Top 25 features (permutation importance)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150)
