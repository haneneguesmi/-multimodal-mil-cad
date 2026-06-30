# ============================================================
# oof_analysis.py   (Notebook 2 / standalone, CPU only)
# The Visual Computer revision -- reviewer points 7 and 8
# Calibration (Brier + reliability), decision curve, age subgroups.
# Uses ONLY existing OOF predictions. No retraining, no torch.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

# ---------------- paths / columns -----------------------------------
INPUT_DIR    = "/kaggle/input/results/publication_outputs/"
DATA_ROOT    = "/kaggle/input/multimodal4cad/MultiD4CAD_Dataset_v1/MultiD4CAD_Dataset_v1/"
CLINICAL_CSV = os.path.join(DATA_ROOT, "clinicalDataWithLabels.CSV")
OUT_DIR      = "/kaggle/working/revision_outputs/"
ID_COL       = "PatientID"
LABEL_COL    = "Label (CAD=1; no CAD=0)"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- load OOF predictions ------------------------------
def load_oof():
    p_all = os.path.join(INPUT_DIR, "oof_all_ablations.csv")
    if os.path.exists(p_all):
        d = pd.read_csv(p_all)
        d["PatientID"] = d["PatientID"].astype(str)
        return d[["PatientID","y_true","prob_full","prob_clinical"]].dropna()
    # fallback: merge full + clinical
    f = pd.read_csv(os.path.join(INPUT_DIR,"oof_full_model.csv"))
    c = pd.read_csv(os.path.join(INPUT_DIR,"oof_clinical_only.csv"))
    f["PatientID"]=f["PatientID"].astype(str); c["PatientID"]=c["PatientID"].astype(str)
    d = f[["PatientID","y_true","prob_full"]].merge(
        c[["PatientID","prob_clinical"]], on="PatientID")
    return d.dropna()

oof = load_oof()
y = oof["y_true"].values.astype(int)
p_full = oof["prob_full"].values.astype(float)
p_clin = oof["prob_clinical"].values.astype(float)
print(f"OOF loaded: n={len(oof)} | CAD+={int(y.sum())}")
print(f"AUC full={roc_auc_score(y,p_full):.3f} | AUC clinical={roc_auc_score(y,p_clin):.3f}")

# ---------------- A) calibration: Brier + reliability ---------------
print("\n== A) CALIBRATION ==")
def reliability(y, p, n_bins=5):
    edges = np.linspace(0,1,n_bins+1); rows=[]
    for i in range(n_bins):
        m = (p>=edges[i]) & (p< edges[i+1] if i<n_bins-1 else p<=edges[i+1])
        if m.sum()>0:
            rows.append({"bin_lo":edges[i],"bin_hi":edges[i+1],
                         "n":int(m.sum()),"pred_mean":float(p[m].mean()),
                         "obs_freq":float(y[m].mean())})
    return pd.DataFrame(rows)

brier_full = brier_score_loss(y, p_full)
brier_clin = brier_score_loss(y, p_clin)
print(f"  Brier full     = {brier_full:.4f}")
print(f"  Brier clinical = {brier_clin:.4f}")
pd.DataFrame([{"model":"full","brier":brier_full},
              {"model":"clinical","brier":brier_clin}]).to_csv(
    os.path.join(OUT_DIR,"rev_calibration.csv"), index=False)
reliability(y,p_full).to_csv(os.path.join(OUT_DIR,"rev_reliability_full.csv"),index=False)
reliability(y,p_clin).to_csv(os.path.join(OUT_DIR,"rev_reliability_clinical.csv"),index=False)

# ---------------- B) decision curve analysis ------------------------
print("\n== B) DECISION CURVE (net benefit) ==")
def net_benefit(y, p, thr):
    pred = (p>=thr).astype(int); n=len(y)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
    return tp/n - fp/n * (thr/(1-thr))
prev = y.mean()
ths = np.round(np.arange(0.05, 0.51, 0.05), 2)
dca_rows=[]
for t in ths:
    nb_full = net_benefit(y,p_full,t)
    nb_clin = net_benefit(y,p_clin,t)
    nb_all  = prev - (1-prev)*(t/(1-t))     # treat-all
    dca_rows.append({"threshold":float(t),"nb_full":nb_full,"nb_clinical":nb_clin,
                     "nb_treat_all":nb_all,"nb_treat_none":0.0})
dca=pd.DataFrame(dca_rows)
dca.to_csv(os.path.join(OUT_DIR,"rev_dca.csv"), index=False)
row20 = dca[dca["threshold"]==0.20].iloc[0]
print(f"  at threshold 0.20: full={row20['nb_full']:.4f} "
      f"clinical={row20['nb_clinical']:.4f} treat-all={row20['nb_treat_all']:.4f}")
# net reduction in false positives per 100 vs treat-all and vs clinical at 0.20
def fp_per100(y,p,thr):
    pred=(p>=thr).astype(int); return 100*int(((pred==1)&(y==0)).sum())/len(y)
print(f"  FP/100 at 0.20: full={fp_per100(y,p_full,0.20):.1f} "
      f"clinical={fp_per100(y,p_clin,0.20):.1f}")

# ---------------- C) age subgroup AUC for FULL ----------------------
print("\n== C) AGE SUBGROUP (FULL model) ==")
try:
    cdf = pd.read_csv(CLINICAL_CSV, sep=";")
    cdf.columns=[c.strip().replace('"','') for c in cdf.columns]
    if ID_COL not in cdf.columns:
        cand=[c for c in cdf.columns if c.lower().strip()==ID_COL.lower().strip()]
        if len(cand)==1: cdf=cdf.rename(columns={cand[0]:ID_COL})
    cdf[ID_COL]=cdf[ID_COL].astype(str)
    AGE_COL=None
    for c in cdf.columns:
        cl=c.strip().lower()
        if (cl=="age" or cl.startswith("age")) and c not in (ID_COL,LABEL_COL):
            AGE_COL=c; break
    if AGE_COL:
        amap=dict(zip(cdf[ID_COL], pd.to_numeric(cdf[AGE_COL],errors="coerce")))
        age=np.array([amap.get(pid,np.nan) for pid in oof["PatientID"]],float)
        med=np.nanmedian(age)
        rows=[]
        for lab,mask in [("age<=median",age<=med),("age>median",age>med)]:
            m=mask & ~np.isnan(age)
            if len(np.unique(y[m]))==2:
                a=roc_auc_score(y[m],p_full[m])
                rows.append({"stratum":lab,"median_age":float(med),
                             "n":int(m.sum()),"auc_full":a})
                print(f"  {lab:12s} (n={int(m.sum()):3d}) AUC_full={a:.3f}")
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"rev_age_subgroup.csv"),index=False)
    else:
        print("  No age column detected in clinical CSV.")
except FileNotFoundError:
    print(f"  {CLINICAL_CSV} not found -> age subgroup skipped.")

print("\nDONE. CSVs in", OUT_DIR)
