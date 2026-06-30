#!/usr/bin/env python3
# ============================================================
# make_synthetic_example.py
# Generates a SYNTHETIC toy dataset that mimics the MultiD4CAD
# directory layout, so the full MIL pipeline can be run and
# reproduced WITHOUT any access to the restricted patient data.
#
# It writes, under ./toy_data/ :
#   clinicalDataWithLabels.CSV                         (sep=";")
#   Epicardial Adipose Tissue Segmentations (118 patients)/<PID>/
#       epicardialDatasetNIFTI.nii
#       epicardialDatasetNIFTI_mask.nii
#   Pericoronaric Adipose Tissue Segmentations (118 patients)/<PID>/
#       coronaricDatasetNIFTI.nii
#       coronaricDatasetNIFTI_mask.nii
#
# The images contain random intensities and a small blob mask, which
# is enough for the 2.5D instance builder. No real anatomy, no PHI.
# ============================================================
import os
import numpy as np
import pandas as pd
import nibabel as nib

# ---- configuration of the toy set -------------------------------
OUT_ROOT   = "toy_data"
N_PATIENTS = 16          # small, both classes
VOL_SHAPE  = (64, 64, 24)  # (H, W, depth) -- depth is the slice axis
SEED       = 0
rng = np.random.default_rng(SEED)

EAT_DIR = "Epicardial Adipose Tissue Segmentations (118 patients)"
PAT_DIR = "Pericoronaric Adipose Tissue Segmentations (118 patients)"

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, EAT_DIR), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, PAT_DIR), exist_ok=True)


def make_volume_and_mask(depth_blob_frac=0.6, blob_radius=8):
    """Random CT-like volume + a blob mask on a subset of slices."""
    H, W, D = VOL_SHAPE
    vol = rng.normal(0.0, 1.0, size=VOL_SHAPE).astype(np.float32)
    mask = np.zeros(VOL_SHAPE, dtype=np.float32)
    # put a circular blob on the central slices so mask area >= 10 px
    n_slices = max(3, int(D * depth_blob_frac))
    start = (D - n_slices) // 2
    yy, xx = np.ogrid[:H, :W]
    for k in range(start, start + n_slices):
        cy = rng.integers(blob_radius, H - blob_radius)
        cx = rng.integers(blob_radius, W - blob_radius)
        r = blob_radius + rng.integers(-2, 3)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        mask[disk, k] = 1.0
        # make the tissue inside the blob slightly brighter
        vol[disk, k] += 2.0
    return vol, mask


def save_nii(arr, path):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), path)


rows = []
for i in range(N_PATIENTS):
    pid = f"P{i+1:03d}"
    label = int(i % 2 == 0)  # alternate classes -> balanced

    # EAT (larger depot) and PAT (smaller depot, contained-like)
    eat_vol, eat_msk = make_volume_and_mask(depth_blob_frac=0.7, blob_radius=10)
    pat_vol, pat_msk = make_volume_and_mask(depth_blob_frac=0.5, blob_radius=6)

    eat_p = os.path.join(OUT_ROOT, EAT_DIR, pid)
    pat_p = os.path.join(OUT_ROOT, PAT_DIR, pid)
    os.makedirs(eat_p, exist_ok=True)
    os.makedirs(pat_p, exist_ok=True)

    save_nii(eat_vol, os.path.join(eat_p, "epicardialDatasetNIFTI.nii"))
    save_nii(eat_msk, os.path.join(eat_p, "epicardialDatasetNIFTI_mask.nii"))
    save_nii(pat_vol, os.path.join(pat_p, "coronaricDatasetNIFTI.nii"))
    save_nii(pat_msk, os.path.join(pat_p, "coronaricDatasetNIFTI_mask.nii"))

    # clinical row: make Age weakly informative so the pipeline is non-trivial
    age = int(rng.normal(58 if label == 1 else 64, 8))
    rows.append({
        "PatientID": pid,
        "Age": age,
        "BMI": round(float(rng.normal(27, 4)), 1),
        "Sex": rng.choice(["M", "F"]),
        "Smoking": rng.choice(["yes", "no"]),
        "Diabetes": rng.choice(["yes", "no"]),
        "Arterial hypertension": rng.choice(["yes", "no"]),
        "Hypercholesterolemia": rng.choice(["yes", "no"]),
        "Family history": rng.choice(["yes", "no"]),
        "Label (CAD=1; no CAD=0)": label,
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_ROOT, "clinicalDataWithLabels.CSV")
df.to_csv(csv_path, sep=";", index=False)

print(f"Synthetic toy dataset written under: {OUT_ROOT}/")
print(f"  patients: {N_PATIENTS} (CAD+ {int(df['Label (CAD=1; no CAD=0)'].sum())}, "
      f"CAD- {int((1-df['Label (CAD=1; no CAD=0)']).sum())})")
print(f"  clinical CSV: {csv_path}")
print(f"  EAT dir: {os.path.join(OUT_ROOT, EAT_DIR)}/<PID>/")
print(f"  PAT dir: {os.path.join(OUT_ROOT, PAT_DIR)}/<PID>/")
print("\nThis is SYNTHETIC data with no real anatomy and no patient information.")
