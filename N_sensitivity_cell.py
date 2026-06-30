# ============================================================
# N_sensitivity_cell.py   (Notebook 3 / the ONLY heavy part)
# Paste as a NEW CELL at the END of your MAIN notebook, after Run All.
# It reuses: CONFIG, df_all, build_cache_dual_2p5d, run_full_model,
#            bootstrap_ci_auc, set_seed  (already defined above).
# It retrains the full model for N=16 and N=32 (N=24 already done).
# Expect a few hours, same order as your ablation study.
# ============================================================
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

orig_N, orig_cache, orig_out = (CONFIG["num_instances"],
                                CONFIG["cache_file"], CONFIG["output_dir"])
cache_dual_24 = cache_dual   # keep N=24 cache in memory

# N=24 from the run you already did
of = pd.read_csv(os.path.join(orig_out, "oof_full_model.csv")).dropna()
y24 = of["y_true"].values.astype(int); p24 = of["prob_full"].values.astype(float)
lo24, hi24 = bootstrap_ci_auc(y24, p24, n_boot=5000)
rows = [{"N":24, "auc":roc_auc_score(y24,p24), "ci_lo":lo24, "ci_hi":hi24,
         "note":"from main run"}]

for Nval in [16, 32]:
    print(f"\n--- N = {Nval}: rebuild cache + retrain ---")
    CONFIG["num_instances"] = Nval
    CONFIG["cache_file"]    = f"/kaggle/working/cache_mil_2_5d_dual_N{Nval}.pkl"
    CONFIG["output_dir"]    = os.path.join(orig_out, f"N{Nval}")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    cache_dual = build_cache_dual_2p5d(df_all)   # global used by the Dataset
    set_seed(42)
    _res, _oof = run_full_model()
    m  = ~_oof["prob_full"].isna().values
    yy = _oof["y_true"].values.astype(int)[m]
    pp = _oof["prob_full"].values.astype(float)[m]
    lo, hi = bootstrap_ci_auc(yy, pp, n_boot=5000)
    rows.append({"N":Nval, "auc":roc_auc_score(yy,pp), "ci_lo":lo, "ci_hi":hi,
                 "note":"retrained"})
    print(f"N={Nval}: AUC={roc_auc_score(yy,pp):.3f} [{lo:.3f}, {hi:.3f}]")

# restore
CONFIG["num_instances"], CONFIG["cache_file"], CONFIG["output_dir"] = \
    orig_N, orig_cache, orig_out
cache_dual = cache_dual_24

dfN = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
print("\nN-sensitivity:"); print(dfN.to_string(index=False))
dfN.to_csv(os.path.join(orig_out, "rev_N_sensitivity.csv"), index=False)
print("saved rev_N_sensitivity.csv")
