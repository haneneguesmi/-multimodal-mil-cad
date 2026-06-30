# ============================================================
# tabular_baselines_aligned_cell.py
# Paste as a NEW CELL at the END of the MAIN notebook, after Run All.
# Reuses: df_all, cache_dual (only to confirm cohort), CONFIG,
#         ClinicalProcessor, bootstrap_ci_auc, bootstrap_delta_auc.
# Computes Logistic / RF / GB / HGB with the SAME protocol as
# run_clinical_only_oof (same df_all filtered by cache, same folds,
# same in-fold ClinicalProcessor). The logistic value here MUST match
# the 0.644 of the paper. One protocol, one value.
# ============================================================
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import roc_auc_score

ID_COL    = CONFIG["patient_id_col"]
LABEL_COL = CONFIG["label_col"]
OUT       = CONFIG["output_dir"]
SEEDS     = CONFIG["seeds"]

# IMPORTANT: use the SAME cohort as the paper (df_all already filtered by cache)
print(f"Cohort used (must be the paper's 117): {len(df_all)} patients")
y_all = df_all[LABEL_COL].values.astype(int)

def oof_tabular(builder, n_seeds=3):
    """OOF probs using the EXACT protocol of run_clinical_only_oof:
       same StratifiedKFold(random_state=42), in-fold ClinicalProcessor."""
    skf = StratifiedKFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=42)
    acc = np.zeros((n_seeds, len(df_all)), float)
    for si in range(n_seeds):
        for tr, va in skf.split(np.zeros(len(df_all)), y_all):
            d_tr = df_all.iloc[tr].reset_index(drop=True)
            d_va = df_all.iloc[va].reset_index(drop=True)
            cp = ClinicalProcessor()
            cp.fit(d_tr, ID_COL, LABEL_COL)
            Xtr, ytr, _ = cp.transform(d_tr, ID_COL, LABEL_COL)
            Xva, yva, _ = cp.transform(d_va, ID_COL, LABEL_COL)
            clf = builder(SEEDS[si])
            clf.fit(Xtr, ytr)
            acc[si, va] = clf.predict_proba(Xva)[:, 1]
    return acc.mean(axis=0)

builders = {
    "Logistic regression": lambda s: LogisticRegression(max_iter=1000, solver="lbfgs"),
    "Random forest":       lambda s: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced",
        random_state=s, n_jobs=-1),
    "Gradient boosting":   lambda s: GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.9,
        random_state=s),
    "Hist grad boosting":  lambda s: HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_iter=400, l2_regularization=1.0,
        random_state=s),
}

print("\n== Tabular baselines (aligned to paper protocol) ==")
rows, probs = [], {}
for name, b in builders.items():
    nseed = 1 if name == "Logistic regression" else len(SEEDS)  # LR deterministic
    p = oof_tabular(b, n_seeds=nseed); probs[name] = p
    auc = roc_auc_score(y_all, p); lo, hi = bootstrap_ci_auc(y_all, p, n_boot=5000)
    rows.append({"model": name, "auc": auc, "ci_lo": lo, "ci_hi": hi})
    print(f"  {name:22s} AUC={auc:.4f} [{lo:.4f}, {hi:.4f}]")

# sanity check: logistic must match the paper's clinical-only OOF
lr_auc = rows[0]["auc"]
print(f"\n[CHECK] Logistic AUC = {lr_auc:.4f}  (paper clinical-only = 0.644). "
      f"{'MATCH' if abs(lr_auc-0.644)<0.01 else 'MISMATCH -> investigate cohort/columns'}")

# paired delta vs FULL model, using the paper's own OOF
of = pd.read_csv(os.path.join(OUT, "oof_full_model.csv"))
of[ID_COL if ID_COL in of.columns else "PatientID"] = \
    of[ID_COL if ID_COL in of.columns else "PatientID"].astype(str)
key = ID_COL if ID_COL in of.columns else "PatientID"
fmap = dict(zip(of[key].astype(str), of["prob_full"].astype(float)))
p_full = np.array([fmap.get(str(pid), np.nan) for pid in df_all[ID_COL].astype(str)], float)
m0 = ~np.isnan(p_full)
print("\n== delta-AUC: FULL minus each tabular baseline (paired) ==")
drows=[]
for name, p in probs.items():
    m = m0 & ~np.isnan(p)
    d = roc_auc_score(y_all[m], p_full[m]) - roc_auc_score(y_all[m], p[m])
    lo, hi, _ = bootstrap_delta_auc(y_all[m], p_full[m], p[m], n_boot=5000)
    sig = "significant" if lo > 0 else "not significant"
    drows.append({"comparison": f"FULL - {name}", "delta_auc": d,
                  "ci_lo": lo, "ci_hi": hi, "significant": sig})
    print(f"  FULL - {name:22s} dAUC={d*100:5.1f}% [{lo*100:5.1f},{hi*100:5.1f}] ({sig})")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev_tabular_baselines_aligned.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "rev_tabular_deltaAUC_aligned.csv"), index=False)
print("\nSaved rev_tabular_baselines_aligned.csv and rev_tabular_deltaAUC_aligned.csv")
