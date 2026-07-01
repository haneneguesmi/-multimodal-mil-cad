# Weakly Supervised Multimodal Attention-Based MIL for CAD Detection

Code accompanying the manuscript *Weakly Supervised Multimodal Attention-Based
Multiple Instance Learning for Coronary Artery Disease Detection from Epicardial
and Pericoronary Adipose Tissue: A Pilot Study* (The Visual Computer).

This repository provides the full training and evaluation pipeline, the revision
analysis scripts, fold assignments, model configuration, and a **synthetic toy
dataset generator** so the pipeline can be reproduced **without access to the
restricted patient data**.

## Data availability

The MultiD4CAD dataset (Prinzi et al., *Scientific Data*, 2025,
doi:10.1038/s41597-025-05743-w) is held under controlled access on Zenodo and is
not redistributed here. To reproduce results on the real data, request access
from the dataset authors and place it under the path expected by `CONFIG`.

To run the pipeline **without** the restricted data, use the synthetic example
described below. It mimics the directory layout and file formats of the real
dataset using random volumes and blob masks. The synthetic data contain **no real
anatomy and no patient information**, and the numerical results obtained on them
are meaningless; the synthetic example exists only to demonstrate that the code
runs end to end.

## Repository contents

| File | Purpose |
| --- | --- |
| `finale.ipynb` | Main pipeline: 2.5D cache, ClinicalProcessor, dual ResNet18 + gated-attention MIL, 5-fold/3-seed training, OOF outputs, ablation study. |
| `make_synthetic_example.py` | Generates the synthetic toy dataset under `toy_data/`. |
| `synthetic_config_patch.py` | Cell to paste after `CONFIG` to switch the notebook to synthetic mode. |
| `clinical_baselines.py` | Nonlinear tabular baselines (RF/GBM/HGB) + age analyses (CPU). |
| `oof_analysis.py` | Calibration, decision curve, age-subgroup analyses from saved OOF predictions (CPU). |
| `N_sensitivity_cell.py` | Instance-count sensitivity (N = 16/24/32). |

## Quick start (synthetic example)

```bash
pip install numpy pandas nibabel scikit-learn torch torchvision tqdm
python make_synthetic_example.py        # creates ./toy_data/
```

Then open `finale.ipynb` and:

1. Run the `CONFIG` cell.
2. Run the `synthetic_config_patch.py` cell (paste it right after `CONFIG`). It
   sets `USE_SYNTHETIC = True`, points `data_root` to `toy_data/`, trains the CNN
   from scratch (no pretrained weights required), and uses a small/fast setting.
3. Run the remaining cells. The pipeline will build the cache, train, and write
   out-of-fold predictions under `toy_outputs/`.

To run on the real dataset, set `USE_SYNTHETIC = False` and restore the original
`CONFIG` paths.

## Expected data layout (real dataset)

Place the real (restricted) dataset under `data/MultiD4CAD/` with this structure,
then set `USE_SYNTHETIC = False`:

```
data/MultiD4CAD/
  clinicalDataWithLabels.CSV                                   # ; separated
  Epicardial Adipose Tissue Segmentations (118 patients)/
    <PatientID>/epicardialDatasetNIFTI.nii
    <PatientID>/epicardialDatasetNIFTI_mask.nii
  Pericoronaric Adipose Tissue Segmentations (118 patients)/
    <PatientID>/coronaricDatasetNIFTI.nii
    <PatientID>/coronaricDatasetNIFTI_mask.nii
```

Pretrained ResNet18 weights (optional) go under `weights/resnet18-f37072fd.pth`.
In synthetic mode no weights are needed; the network trains from scratch.
The synthetic example writes its own `toy_data/` and ignores these paths.

## Notes

- Fold assignments are determined by `StratifiedKFold(n_splits=..., shuffle=True,
  random_state=42)` and are therefore reproducible.
- The clinical preprocessing (`ClinicalProcessor`) auto-detects numeric vs
  categorical columns and applies in-fold standardization and one-hot encoding.
- `synthetic_config_patch.py` also defines `best_threshold_by_f1`, a small helper
  used by the ablation table.
