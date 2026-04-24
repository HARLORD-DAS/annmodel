"""
==============================================================
  Phase 1 — Step 1: Dataset Acquisition
  ISLES 2022 (Ischemic Stroke Lesion Segmentation)
==============================================================

ISLES 2022 is the best public dataset for this project:
  - 400 multi-center MRI cases (DWI + ADC maps)
  - Expert lesion segmentation masks included
  - Associated clinical metadata

Download options:
  A) Official: https://isles22.grand-challenge.org/  (free registration)
  B) Synapse platform: syn27225181
  C) PhysioNet: https://physionet.org/content/isles-2022/

This script handles:
  - Directory setup
  - Synthetic data generation (for development/testing before real data arrives)
  - Data integrity checks
  - Folder structure validation
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Load config
# ──────────────────────────────────────────────
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# ISLES 2022 expected folder structure
# ──────────────────────────────────────────────
ISLES_STRUCTURE = """
Expected ISLES 2022 folder structure after download:

data/raw/ISLES2022/
├── derivatives/
│   └── sub-strokeXXXX/
│       └── ses-XXXX/
│           └── sub-strokeXXXX_ses-XXXX_lesion-msk.nii.gz  ← segmentation mask
├── sub-strokeXXXX/
│   └── ses-XXXX/
│       └── dwi/
│           ├── sub-strokeXXXX_ses-XXXX_dwi.nii.gz         ← DWI volume
│           └── sub-strokeXXXX_ses-XXXX_adc.nii.gz         ← ADC map
└── participants.tsv                                         ← clinical metadata
"""


# ──────────────────────────────────────────────
# Synthetic data generator (development mode)
# ──────────────────────────────────────────────
def generate_synthetic_dataset(n_subjects=200, output_dir="data/raw/synthetic", seed=42):
    """
    Generates realistic synthetic MRI volumes + clinical data for development.
    Use this while waiting for ISLES 2022 access, or to test the pipeline end-to-end.

    Synthetic MRI:
      - 3D NIfTI volumes (64x64x32) with simulated stroke lesions
      - DWI: high signal in lesion region (restricted diffusion)
      - ADC: low signal in lesion region (cytotoxic edema)

    Synthetic clinical:
      - NIHSS scores drawn from realistic distributions per severity class
      - Comorbidities with clinically realistic prevalence rates
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/mri", exist_ok=True)

    print(f"\n{'='*55}")
    print("  Generating synthetic dataset for development")
    print(f"  Subjects: {n_subjects}   Output: {output_dir}")
    print(f"{'='*55}\n")

    # Severity thresholds (NIHSS-based)
    # mild < 7, moderate 7–15, severe > 15
    severity_probs = {"mild": 0.45, "moderate": 0.35, "severe": 0.20}
    severity_nihss = {
        "mild":     lambda: np.random.randint(1, 7),
        "moderate": lambda: np.random.randint(7, 16),
        "severe":   lambda: np.random.randint(16, 35),
    }

    records = []

    for i in tqdm(range(n_subjects), desc="Generating subjects"):
        subj_id = f"sub-synth{i+1:04d}"

        # ── Assign severity ──
        severity = np.random.choice(
            list(severity_probs.keys()),
            p=list(severity_probs.values())
        )
        nihss_baseline = severity_nihss[severity]()

        # ── Simulate lesion ──
        volume_shape = (64, 64, 32)
        lesion_vol_frac = {
            "mild":     np.random.uniform(0.002, 0.02),
            "moderate": np.random.uniform(0.02, 0.08),
            "severe":   np.random.uniform(0.08, 0.20),
        }[severity]

        # Create lesion mask (ellipsoidal lesion placed in plausible stroke territory)
        mask = np.zeros(volume_shape, dtype=np.float32)
        cx = np.random.randint(20, 44)
        cy = np.random.randint(20, 44)
        cz = np.random.randint(8, 24)
        rx = int((lesion_vol_frac * 64 * 64 * 32) ** (1/3) * 1.5)
        rx = max(2, min(rx, 12))
        ry = rx + np.random.randint(-2, 3)
        rz = max(1, rx - 2)

        for x in range(max(0, cx-rx-2), min(64, cx+rx+2)):
            for y in range(max(0, cy-ry-2), min(64, cy+ry+2)):
                for z in range(max(0, cz-rz-1), min(32, cz+rz+1)):
                    if ((x-cx)/rx)**2 + ((y-cy)/ry)**2 + ((z-cz)/rz)**2 <= 1:
                        mask[x, y, z] = 1.0

        # ── Simulate DWI volume ──
        dwi = np.random.normal(300, 60, volume_shape).astype(np.float32)
        # Lesion: restricted diffusion → hyperintense on DWI
        dwi += mask * np.random.uniform(400, 800)
        # Add Gaussian noise
        dwi += np.random.normal(0, 15, volume_shape)
        dwi = np.clip(dwi, 0, None)

        # ── Simulate ADC map ──
        adc = np.random.normal(800, 120, volume_shape).astype(np.float32)
        # Lesion: cytotoxic edema → hypointense on ADC
        adc -= mask * np.random.uniform(400, 600)
        adc += np.random.normal(0, 20, volume_shape)
        adc = np.clip(adc, 0, None)

        # ── Save NIfTI files ──
        affine = np.eye(4)  # Identity affine (isotropic 1mm voxels)
        nib.save(nib.Nifti1Image(dwi, affine), f"{output_dir}/mri/{subj_id}_dwi.nii.gz")
        nib.save(nib.Nifti1Image(adc, affine), f"{output_dir}/mri/{subj_id}_adc.nii.gz")
        nib.save(nib.Nifti1Image(mask, affine), f"{output_dir}/mri/{subj_id}_mask.nii.gz")

        # ── Simulate clinical data ──
        age = int(np.clip(np.random.normal(65, 12), 30, 95))
        gender = np.random.randint(0, 2)
        onset_hours = np.random.exponential(4.5)
        onset_hours = min(onset_hours, 24.0)

        hypertension = 1 if np.random.random() < 0.62 else 0
        diabetes = 1 if np.random.random() < 0.28 else 0
        af = 1 if np.random.random() < 0.22 else 0
        prior_stroke = 1 if np.random.random() < 0.14 else 0
        systolic_bp = int(np.random.normal(155 if hypertension else 130, 20))
        glucose = round(np.random.normal(7.2 if diabetes else 5.8, 1.5), 1)

        lesion_vol_ml = float(np.sum(mask) * 1.0)  # 1mm³ voxels → mm³, /1000 = ml
        lesion_vol_ml = round(lesion_vol_ml / 1000, 2)

        # ── Simulate 90-day NIHSS (outcome) ──
        # Improvement probability by severity
        improvement_p = {"mild": 0.80, "moderate": 0.55, "severe": 0.25}[severity]
        if np.random.random() < improvement_p:
            # Improvement amount depends on severity
            if severity == "mild":
                improvement = np.random.randint(1, min(3, nihss_baseline + 1))
            elif severity == "moderate":
                improvement = np.random.randint(2, min(8, nihss_baseline + 1))
            else:  # severe
                improvement = np.random.randint(1, min(5, nihss_baseline + 1))
            nihss_day90 = max(0, nihss_baseline - improvement)
        else:
            nihss_day90 = nihss_baseline + np.random.randint(0, 8)

        # ── Lesion location (simplified) ──
        location_map = {0: "frontal", 1: "parietal", 2: "temporal", 3: "occipital", 4: "basal_ganglia", 5: "cerebellum"}
        lesion_location = location_map[np.random.randint(0, 6)]

        records.append({
            "subject_id": subj_id,
            "age": age,
            "gender": gender,
            "nihss_baseline": nihss_baseline,
            "nihss_day90": nihss_day90,
            "severity_label": severity,
            "onset_to_door_hours": round(onset_hours, 2),
            "hypertension": hypertension,
            "diabetes": diabetes,
            "atrial_fibrillation": af,
            "prior_stroke": prior_stroke,
            "systolic_bp": systolic_bp,
            "glucose_admission": glucose,
            "lesion_volume_ml": lesion_vol_ml,
            "lesion_location": lesion_location,
            "dwi_path": f"mri/{subj_id}_dwi.nii.gz",
            "adc_path": f"mri/{subj_id}_adc.nii.gz",
            "mask_path": f"mri/{subj_id}_mask.nii.gz",
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{output_dir}/participants.csv", index=False)

    print(f"\n✓ Generated {n_subjects} synthetic subjects")
    print(f"✓ MRI volumes saved to {output_dir}/mri/")
    print(f"✓ Clinical data saved to {output_dir}/participants.csv")
    print(f"\nClass distribution:")
    print(df["severity_label"].value_counts())
    print(f"\nNIHSS baseline stats:")
    print(df["nihss_baseline"].describe().round(2))

    return df


# ──────────────────────────────────────────────
# ISLES 2022 real data parser
# ──────────────────────────────────────────────
def parse_isles2022(isles_root="data/raw/ISLES2022"):
    """
    Parse real ISLES 2022 dataset into a unified DataFrame.
    Run this after downloading from grand-challenge.org.
    """
    isles_root = Path(isles_root)
    participants_file = isles_root / "participants.tsv"

    if not participants_file.exists():
        raise FileNotFoundError(
            f"\nISLES 2022 not found at {isles_root}\n"
            f"Please download from: https://isles22.grand-challenge.org/\n"
            f"Or run generate_synthetic_dataset() for development.\n"
            f"{ISLES_STRUCTURE}"
        )

    df = pd.read_csv(participants_file, sep="\t")
    print(f"✓ Loaded ISLES 2022: {len(df)} subjects")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing MRI paths"):
        subj = row["participant_id"]
        sess = row.get("session_id", "ses-01")

        mri_dir = isles_root / subj / sess / "dwi"
        mask_dir = isles_root / "derivatives" / subj / sess

        dwi_path = list(mri_dir.glob("*_dwi.nii.gz"))
        adc_path = list(mri_dir.glob("*_adc.nii.gz"))
        mask_path = list(mask_dir.glob("*_lesion-msk.nii.gz"))

        record = row.to_dict()
        record["dwi_path"] = str(dwi_path[0]) if dwi_path else None
        record["adc_path"] = str(adc_path[0]) if adc_path else None
        record["mask_path"] = str(mask_path[0]) if mask_path else None

        # Derive severity from NIHSS if present
        if "nihss_baseline" in record:
            nihss = record["nihss_baseline"]
            if nihss < 7:
                record["severity_label"] = "mild"
            elif nihss <= 15:
                record["severity_label"] = "moderate"
            else:
                record["severity_label"] = "severe"

        records.append(record)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Data integrity checker
# ──────────────────────────────────────────────
def check_data_integrity(df, data_dir="data/raw/synthetic"):
    """Validate that all MRI files exist and are readable."""
    print("\n── Running data integrity checks ──")
    issues = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating files"):
        for col in ["dwi_path", "adc_path", "mask_path"]:
            if col not in df.columns:
                continue
            path = os.path.join(data_dir, row[col]) if not os.path.isabs(str(row[col])) else row[col]
            if not os.path.exists(path):
                issues.append(f"Missing: {path}")
                continue
            try:
                img = nib.load(path)
                _ = img.get_fdata()
            except Exception as e:
                issues.append(f"Corrupt: {path} — {e}")

    if issues:
        print(f"\n⚠ Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
    else:
        print(f"✓ All {len(df)} subjects passed integrity check")

    return issues


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stroke ANN — Dataset Acquisition")
    parser.add_argument("--mode", choices=["synthetic", "isles2022"], default="synthetic",
                        help="Use synthetic data (dev) or real ISLES 2022")
    parser.add_argument("--n_subjects", type=int, default=200,
                        help="Number of synthetic subjects to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "synthetic":
        df = generate_synthetic_dataset(
            n_subjects=args.n_subjects,
            output_dir="data/raw/synthetic",
            seed=args.seed
        )
        issues = check_data_integrity(df, data_dir="data/raw/synthetic")
        df.to_csv("data/raw/synthetic/participants_validated.csv", index=False)
        print(f"\n✓ Dataset ready. Run next: python src/preprocessing/mri_preprocessor.py")

    elif args.mode == "isles2022":
        print(ISLES_STRUCTURE)
        df = parse_isles2022("data/raw/ISLES2022")
        issues = check_data_integrity(df, data_dir="data/raw/ISLES2022")
        df.to_csv("data/raw/isles2022_index.csv", index=False)
        print(f"\n✓ ISLES 2022 indexed. Run next: python src/preprocessing/mri_preprocessor.py")
