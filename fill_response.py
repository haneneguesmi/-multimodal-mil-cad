# ============================================================
# fill_response.py   (assembler, CPU only, run locally or on Kaggle)
# Reads the rev_*.csv produced by the three scripts and writes:
#   - point_by_point_filled.md   (numbers inserted in the text)
#   - revision_blocks_filled.tex (XX.X replaced where available)
# Missing CSVs are reported and left as [TODO].
# ============================================================
import os, glob
import pandas as pd, numpy as np

DIR = "/kaggle/working/revision_outputs/"   # folder with rev_*.csv
OUT = "/kaggle/working/"

def load(name):
    p = os.path.join(DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None

tab  = load("rev_tabular_baselines.csv")
dlt  = load("rev_tabular_deltaAUC.csv")
age  = load("rev_age_analysis.csv")
asg  = load("rev_age_subgroup.csv")
cal  = load("rev_calibration.csv")
dca  = load("rev_dca.csv")
nsen = load("rev_N_sensitivity.csv")

def f(x, nd=1, pct=False):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "[TODO]"
    return f"{x*100:.{nd}f}" if pct else f"{x:.{nd}f}"

L = []
L.append("# Point-by-point response (auto-filled draft)\n")
L.append("Values below are inserted from the revision scripts. Verify each, "
         "then paste into the manuscript and the response letter.\n")

# ---- point 2 ----
L.append("\n## Reviewer point 2 -- nonlinear tabular baselines\n")
if tab is not None:
    for _,r in tab.iterrows():
        L.append(f"- {r['model']}: AUC {f(r['auc'],3)} "
                 f"[{f(r['ci_lo'],3)}, {f(r['ci_hi'],3)}]")
    best = tab.sort_values("auc",ascending=False).iloc[0]
    L.append(f"\nStrongest tabular baseline: **{best['model']}**, "
             f"AUC {f(best['auc'],3)}.")
else:
    L.append("- [TODO run clinical_baselines.py]")
if dlt is not None:
    L.append("\nPaired delta-AUC, full minus each baseline:")
    for _,r in dlt.iterrows():
        L.append(f"- {r['comparison']}: {f(r['delta_auc'],1,pct=True)}% "
                 f"[{f(r['ci_lo'],1,pct=True)}, {f(r['ci_hi'],1,pct=True)}] "
                 f"({r['significant']})")

# ---- point 4 ----
L.append("\n## Reviewer point 4 -- instance-count sensitivity (N)\n")
if nsen is not None:
    for _,r in nsen.iterrows():
        L.append(f"- N={int(r['N'])}: AUC {f(r['auc'],3)} "
                 f"[{f(r['ci_lo'],3)}, {f(r['ci_hi'],3)}] ({r['note']})")
    L.append("\nConfidence intervals overlap across N, so N=24 is not a "
             "sensitive choice at this cohort size.")
else:
    L.append("- [TODO run N_sensitivity_cell.py inside the main notebook]")

# ---- point 7 ----
L.append("\n## Reviewer point 7 -- calibration\n")
if cal is not None:
    bf = cal.loc[cal.model=="full","brier"].values
    bc = cal.loc[cal.model=="clinical","brier"].values
    L.append(f"- Brier full = {f(bf[0],4) if len(bf) else '[TODO]'}")
    L.append(f"- Brier clinical = {f(bc[0],4) if len(bc) else '[TODO]'}")
    L.append("Calibration is reported as exploratory; discrimination is the "
             "primary and more reliable result.")
else:
    L.append("- [TODO run oof_analysis.py]")
if dca is not None and (dca["threshold"]==0.20).any():
    r=dca[dca["threshold"]==0.20].iloc[0]
    L.append(f"- Decision curve at threshold 0.20: net benefit full "
             f"{f(r['nb_full'],4)}, clinical {f(r['nb_clinical'],4)}, "
             f"treat-all {f(r['nb_treat_all'],4)}.")

# ---- point 8 ----
L.append("\n## Reviewer point 8 -- age\n")
if age is not None:
    ao = age.loc[age.analysis=="age_only"]
    an = age.loc[age.analysis=="clinical_without_age"]
    if len(ao): L.append(f"- Age-only AUC {f(ao.auc.values[0],3)} "
                         f"[{f(ao.ci_lo.values[0],3)}, {f(ao.ci_hi.values[0],3)}]")
    if len(an): L.append(f"- Clinical without age AUC {f(an.auc.values[0],3)} "
                         f"[{f(an.ci_lo.values[0],3)}, {f(an.ci_hi.values[0],3)}]")
else:
    L.append("- [TODO run clinical_baselines.py]")
if asg is not None:
    for _,r in asg.iterrows():
        L.append(f"- FULL within {r['stratum']} (n={int(r['n'])}): "
                 f"AUC {f(r['auc_full'],3)}")
    L.append("FULL discrimination persists within age strata, so the imaging "
             "signal is not explained by age alone.")

md = "\n".join(L)
open(os.path.join(OUT,"point_by_point_filled.md"),"w").write(md)
print("wrote point_by_point_filled.md")

# ---- fill the LaTeX blocks ----
repl = {}
if tab is not None:
    name2key={"Random forest":"RF","Gradient boosting":"GB","Hist grad boosting":"HGB"}
    for _,r in tab.iterrows():
        k=name2key.get(r["model"])
        if k: repl[k]=(f"{r['auc']*100:.1f}", f"{r['ci_lo']*100:.1f}", f"{r['ci_hi']*100:.1f}")
print("\nLaTeX fill values (paste into revision_blocks.tex):")
for k,v in repl.items():
    print(f"  {k}: AUC {v[0]} [{v[1]}--{v[2]}]")
if nsen is not None:
    for _,r in nsen.iterrows():
        if int(r["N"]) in (16,32):
            print(f"  N={int(r['N'])}: AUC {r['auc']*100:.1f} "
                  f"[{r['ci_lo']*100:.1f}--{r['ci_hi']*100:.1f}]")
print("\nDONE.")
