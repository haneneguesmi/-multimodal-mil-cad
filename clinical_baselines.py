# ============================================================
# clinical_baselines.py   (Notebook 1 / standalone, CPU only)
# The Visual Computer revision -- reviewer points 2 and 8
# RF / GBM / HGB / Logistic + age analyses, OOF + bootstrap CI
# No GPU, no deep learning, no cache. Runs in 2-3 minutes.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import roc_auc_score

# ---------------- paths / columns (edit only if needed) -------------
DATA_ROOT    = "/kaggle/input/multimodal4cad/MultiD4CAD_Dataset_v1/MultiD4CAD_Dataset_v1/"
CLINICAL_CSV = os.path.join(DATA_ROOT, "clinicalDataWithLabels.CSV")
OOF_FULL_CSV = "/kaggle/input/results/publication_outputs/oof_full_model.csv"
OUT_DIR      = "/kaggle/working/revision_outputs/"
ID_COL       = "PatientID"
LABEL_COL    = "Label (CAD=1; no CAD=0)"
SEEDS        = [42, 43, 44]
N_FOLDS      = 5
N_BOOT       = 5000
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- load clinical CSV (same conventions as notebook) --
df = pd.read_csv(CLINICAL_CSV, sep=";")
df.columns = [c.strip().replace('"', '') for c in df.columns]
if ID_COL not in df.columns:
    cand = [c for c in df.columns if c.lower().strip() == ID_COL.lower().strip()]
    if len(cand) == 1:
        df = df.rename(columns={cand[0]: ID_COL})
if LABEL_COL not in df.columns:
    for c in df.columns:
        if "CAD" in c.upper() and "LABEL" in c.upper():
            df = df.rename(columns={c: LABEL_COL}); break
df[ID_COL] = df[ID_COL].astype(str)
df[LABEL_COL] = df[LABEL_COL].apply(lambda v: int(float(v))).astype(int)
y_all = df[LABEL_COL].values.astype(int)
print(f"Loaded {len(df)} patients | CAD+ {int(y_all.sum())} | CAD- {int((1-y_all).sum())}")
print("Columns:", list(df.columns))

# ---------------- in-fold preprocessing (matches ClinicalProcessor) -
def fit_transform_fold(d_tr, d_va):
    Xtr_raw = d_tr.drop(columns=[ID_COL, LABEL_COL], errors="ignore")
    num_cols, cat_cols = [], []
    for c in Xtr_raw.columns:
        s = pd.to_numeric(Xtr_raw[c], errors="coerce")
        (num_cols if np.isfinite(s.values).mean() >= 0.8 else cat_cols).append(c)

    def num_block(d):
        if not num_cols: return np.zeros((len(d), 0), np.float32)
        M = d[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
        return M.values.astype(np.float32)
    def cat_block_raw(d):
        if not cat_cols: return None
        return d[cat_cols].astype(str).fillna("NA").values

    sc = StandardScaler().fit(num_block(d_tr)) if num_cols else None
    ohe = (OneHotEncoder(handle_unknown="ignore", sparse_output=False)
           .fit(cat_block_raw(d_tr))) if cat_cols else None

    def tf(d):
        xn = sc.transform(num_block(d)) if sc is not None else np.zeros((len(d),0),np.float32)
        xc = ohe.transform(cat_block_raw(d)) if ohe is not None else np.zeros((len(d),0),np.float32)
        return np.concatenate([xn.astype(np.float32), xc.astype(np.float32)], axis=1)
    return tf(d_tr), tf(d_va)

# ---------------- OOF runner (3-seed ensemble) ----------------------
def oof_predict(df_in, builder, n_seeds=3):
    yv = df_in[LABEL_COL].values.astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    acc = np.zeros((n_seeds, len(df_in)), float)
    for si in range(n_seeds):
        for tr, va in skf.split(np.zeros(len(df_in)), yv):
            d_tr = df_in.iloc[tr].reset_index(drop=True)
            d_va = df_in.iloc[va].reset_index(drop=True)
            Xtr, Xva = fit_transform_fold(d_tr, d_va)
            clf = builder(SEEDS[si]); clf.fit(Xtr, yv[tr])
            acc[si, va] = clf.predict_proba(Xva)[:, 1]
    return acc.mean(axis=0)

def boot_ci_auc(y, p, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed); n=len(y); a=[]
    for _ in range(n_boot):
        idx = rng.integers(0,n,n)
        if len(np.unique(y[idx]))<2: continue
        a.append(roc_auc_score(y[idx], p[idx]))
    a=np.sort(a); return float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

def boot_delta(y, pf, pb, n_boot=N_BOOT, seed=42):
    rng=np.random.default_rng(seed); n=len(y); d=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        if len(np.unique(y[idx]))<2: continue
        d.append(roc_auc_score(y[idx],pf[idx])-roc_auc_score(y[idx],pb[idx]))
    d=np.sort(d); return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

builders = {
    "Logistic regression": lambda s: LogisticRegression(max_iter=2000),
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

# ---------------- A) tabular baselines ------------------------------
print("\n== A) TABULAR BASELINES (OOF) ==")
rows, probs = [], {}
for name, b in builders.items():
    p = oof_predict(df, b, len(SEEDS)); probs[name] = p
    auc = roc_auc_score(y_all, p); lo, hi = boot_ci_auc(y_all, p)
    rows.append({"model": name, "auc": auc, "ci_lo": lo, "ci_hi": hi})
    print(f"  {name:22s} AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]")
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"rev_tabular_baselines.csv"), index=False)

# delta vs FULL model (from saved OOF)
try:
    of = pd.read_csv(OOF_FULL_CSV); of[ID_COL if ID_COL in of.columns else "PatientID"] = \
        of[ID_COL if ID_COL in of.columns else "PatientID"].astype(str)
    key = ID_COL if ID_COL in of.columns else "PatientID"
    fmap = dict(zip(of[key].astype(str), of["prob_full"].astype(float)))
    p_full = np.array([fmap.get(pid, np.nan) for pid in df[ID_COL]], float)
    m0 = ~np.isnan(p_full)
    print("\n== delta-AUC: FULL minus each tabular baseline ==")
    drows=[]
    for name,p in probs.items():
        m=m0 & ~np.isnan(p)
        d=roc_auc_score(y_all[m],p_full[m])-roc_auc_score(y_all[m],p[m])
        lo,hi=boot_delta(y_all[m],p_full[m],p[m])
        sig="significant" if lo>0 else "not significant"
        drows.append({"comparison":f"FULL - {name}","delta_auc":d,"ci_lo":lo,"ci_hi":hi,"significant":sig})
        print(f"  FULL - {name:22s} dAUC={d*100:5.1f}% [{lo*100:5.1f},{hi*100:5.1f}] ({sig})")
    pd.DataFrame(drows).to_csv(os.path.join(OUT_DIR,"rev_tabular_deltaAUC.csv"), index=False)
except FileNotFoundError:
    print(f"[warn] {OOF_FULL_CSV} not found -> skipped delta-AUC. Set OOF_FULL_CSV.")

# ---------------- B) age analyses -----------------------------------
print("\n== B) AGE ANALYSES ==")
AGE_COL=None
for c in df.columns:
    cl=c.strip().lower()
    if (cl=="age" or cl.startswith("age")) and c not in (ID_COL,LABEL_COL):
        AGE_COL=c; break
print("Detected age column:", AGE_COL)
if AGE_COL:
    d_age = df[[ID_COL, AGE_COL, LABEL_COL]].copy()
    p_age = oof_predict(d_age, lambda s: LogisticRegression(max_iter=2000), len(SEEDS))
    a=roc_auc_score(y_all,p_age); lo,hi=boot_ci_auc(y_all,p_age)
    print(f"  Age-only            AUC={a:.3f} [{lo:.3f}, {hi:.3f}]")
    d_noage = df.drop(columns=[AGE_COL]).copy()
    p_noage = oof_predict(d_noage, lambda s: LogisticRegression(max_iter=2000), len(SEEDS))
    a2=roc_auc_score(y_all,p_noage); lo2,hi2=boot_ci_auc(y_all,p_noage)
    print(f"  Clinical w/o age    AUC={a2:.3f} [{lo2:.3f}, {hi2:.3f}]")
    pd.DataFrame([
        {"analysis":"age_only","auc":a,"ci_lo":lo,"ci_hi":hi},
        {"analysis":"clinical_without_age","auc":a2,"ci_lo":lo2,"ci_hi":hi2},
    ]).to_csv(os.path.join(OUT_DIR,"rev_age_analysis.csv"), index=False)
else:
    print("  No age column detected -> set AGE_COL manually.")

print("\nDONE. CSVs in", OUT_DIR)
