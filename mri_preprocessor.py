"""
==============================================================
  Phase 1 — Step 2: MRI Preprocessing Pipeline
  Handles DWI, ADC volumes → normalized feature arrays
==============================================================

Pipeline:
  1. Load NIfTI volumes (DWI, ADC, lesion mask)
  2. Resample to uniform shape (64×64×32)
  3. Intensity normalization (z-score within brain mask)
  4. Extract radiomics features per subject
  5. Save processed arrays + feature CSV
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import warnings
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
import yaml
import json
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# 1. Load MRI volume
# ──────────────────────────────────────────────
def load_nifti(path):
    """Load a NIfTI file and return (data_array, affine, header)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


# ──────────────────────────────────────────────
# 2. Resample to target shape
# ──────────────────────────────────────────────
def resample_volume(volume, target_shape=(64, 64, 32), order=1):
    """
    Resample a 3D volume to target_shape using zoom.
    order=1 → trilinear interpolation (good for continuous volumes)
    order=0 → nearest neighbour (use for binary masks)
    """
    current_shape = np.array(volume.shape[:3])
    target = np.array(target_shape)
    zoom_factors = target / current_shape

    # Handle 4D volumes (e.g. DWI with multiple b-values) — take b=1000 volume
    if volume.ndim == 4:
        volume = volume[..., -1]

    resampled = zoom(volume, zoom_factors, order=order)
    return resampled.astype(np.float32)


# ──────────────────────────────────────────────
# 3. Intensity normalization
# ──────────────────────────────────────────────
def normalize_intensity(volume, mask=None, method="z_score"):
    """
    Normalize MRI intensity within the brain mask.

    z_score:    (x - mean) / std  — standard for MRI
    min_max:    (x - min) / (max - min)
    percentile: clip to [1, 99] percentile then min_max
    """
    if mask is not None and mask.sum() > 0:
        brain_voxels = volume[mask > 0]
    else:
        brain_voxels = volume[volume > 0]

    if method == "z_score":
        mu = brain_voxels.mean()
        sigma = brain_voxels.std() + 1e-8
        normalized = (volume - mu) / sigma
    elif method == "min_max":
        vmin, vmax = brain_voxels.min(), brain_voxels.max()
        normalized = (volume - vmin) / (vmax - vmin + 1e-8)
    elif method == "percentile":
        p1, p99 = np.percentile(brain_voxels, [1, 99])
        volume_clipped = np.clip(volume, p1, p99)
        normalized = (volume_clipped - p1) / (p99 - p1 + 1e-8)
    else:
        normalized = volume.copy()

    return normalized.astype(np.float32)


# ──────────────────────────────────────────────
# 4. Radiomics feature extraction
# ──────────────────────────────────────────────
def extract_radiomics(dwi, adc, mask, subject_id="unknown"):
    """
    Extract clinically meaningful radiomics features from the lesion mask.

    Features extracted:
      - Lesion volume (ml)
      - Mean / min / max / std of DWI signal in lesion
      - Mean / min / max / std of ADC signal in lesion
      - Penumbra ratio (approximated as perilesional tissue ratio)
      - Lesion asymmetry (left vs right hemisphere)
      - Relative lesion volume (lesion / total brain)
    """
    features = {"subject_id": subject_id}

    lesion_voxels_dwi = dwi[mask > 0]
    lesion_voxels_adc = adc[mask > 0]
    total_voxels = np.sum(mask > 0)

    # Lesion volume (assuming 1mm³ voxels → convert to mL)
    features["lesion_volume_ml"] = round(float(total_voxels / 1000), 4)
    features["lesion_voxel_count"] = int(total_voxels)

    # DWI stats in lesion
    if len(lesion_voxels_dwi) > 0:
        features["dwi_mean_lesion"] = float(np.mean(lesion_voxels_dwi))
        features["dwi_std_lesion"] = float(np.std(lesion_voxels_dwi))
        features["dwi_max_lesion"] = float(np.max(lesion_voxels_dwi))
        features["dwi_min_lesion"] = float(np.min(lesion_voxels_dwi))
    else:
        for k in ["dwi_mean_lesion", "dwi_std_lesion", "dwi_max_lesion", "dwi_min_lesion"]:
            features[k] = 0.0

    # ADC stats in lesion (low ADC = acute infarct core)
    if len(lesion_voxels_adc) > 0:
        features["adc_mean_lesion"] = float(np.mean(lesion_voxels_adc))
        features["adc_std_lesion"] = float(np.std(lesion_voxels_adc))
        features["adc_min_lesion"] = float(np.min(lesion_voxels_adc))
        features["adc_max_lesion"] = float(np.max(lesion_voxels_adc))
    else:
        for k in ["adc_mean_lesion", "adc_std_lesion", "adc_min_lesion", "adc_max_lesion"]:
            features[k] = 0.0

    # Penumbra ratio: perilesional zone vs core
    # Dilate mask by ~3 voxels to approximate ischaemic penumbra
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask > 0, iterations=3).astype(np.float32)
    penumbra_mask = (dilated - (mask > 0)).clip(0, 1)
    penumbra_voxels = np.sum(penumbra_mask)
    core_voxels = max(total_voxels, 1)
    features["penumbra_ratio"] = round(float(penumbra_voxels / core_voxels), 4)

    # Hemispheric asymmetry (mid-sagittal split at x = volume_width // 2)
    mid = mask.shape[0] // 2
    left_lesion = float(np.sum(mask[:mid, :, :] > 0))
    right_lesion = float(np.sum(mask[mid:, :, :] > 0))
    total_lesion = left_lesion + right_lesion + 1e-8
    features["lesion_laterality"] = round((right_lesion - left_lesion) / total_lesion, 4)

    # Relative lesion volume (fraction of total brain volume)
    brain_voxels = float(np.sum(dwi > 0))
    features["relative_lesion_volume"] = round(
        float(total_voxels) / max(brain_voxels, 1), 6
    )

    # Superior-inferior center of mass (normalized z-position)
    if total_voxels > 0:
        z_positions = np.where(mask > 0)[2]
        features["lesion_com_z"] = round(float(z_positions.mean()) / mask.shape[2], 4)
    else:
        features["lesion_com_z"] = 0.0

    return features


# ──────────────────────────────────────────────
# 5. Full subject preprocessing
# ──────────────────────────────────────────────
def preprocess_subject(row, data_dir, target_shape=(64, 64, 32), norm_method="z_score"):
    """
    Full preprocessing pipeline for one subject.
    Returns: (dwi_array, adc_array, mask_array, radiomics_dict)
    """
    subj_id = row["subject_id"]

    def resolve(col):
        p = str(row[col])
        if not os.path.isabs(p):
            return os.path.join(data_dir, p)
        return p

    try:
        dwi_raw, affine, _ = load_nifti(resolve("dwi_path"))
        adc_raw, _, _ = load_nifti(resolve("adc_path"))
        mask_raw, _, _ = load_nifti(resolve("mask_path"))
    except Exception as e:
        print(f"\n⚠ Failed to load {subj_id}: {e}")
        return None

    # Resample
    dwi_rs = resample_volume(dwi_raw, target_shape, order=1)
    adc_rs = resample_volume(adc_raw, target_shape, order=1)
    mask_rs = resample_volume(mask_raw, target_shape, order=0)
    mask_rs = (mask_rs > 0.5).astype(np.float32)

    # Normalize
    dwi_norm = normalize_intensity(dwi_rs, mask_rs, method=norm_method)
    adc_norm = normalize_intensity(adc_rs, mask_rs, method=norm_method)

    # Radiomics (on normalized volumes)
    rad = extract_radiomics(dwi_norm, adc_norm, mask_rs, subject_id=subj_id)

    return dwi_norm, adc_norm, mask_rs, rad


# ──────────────────────────────────────────────
# 6. Batch preprocessing pipeline
# ──────────────────────────────────────────────
def run_mri_preprocessing(
    participants_csv,
    data_dir,
    output_dir="data/processed",
    target_shape=(64, 64, 32),
    norm_method="z_score"
):
    """
    Process all subjects: resample, normalize, extract radiomics.
    Saves:
      - data/processed/mri/  → .npy files per subject (DWI, ADC stacked)
      - data/processed/radiomics.csv  → radiomics feature table
    """
    df = pd.read_csv(participants_csv)
    os.makedirs(f"{output_dir}/mri", exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  MRI Preprocessing Pipeline")
    print(f"  Subjects: {len(df)}   Target shape: {target_shape}")
    print(f"  Normalization: {norm_method}")
    print(f"{'='*55}\n")

    radiomics_records = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        result = preprocess_subject(row, data_dir, target_shape, norm_method)

        if result is None:
            failed.append(row["subject_id"])
            continue

        dwi_norm, adc_norm, mask_rs, rad = result

        # Stack DWI + ADC into 2-channel array: shape (64, 64, 32, 2)
        mri_stack = np.stack([dwi_norm, adc_norm], axis=-1)
        np.save(f"{output_dir}/mri/{row['subject_id']}_mri.npy", mri_stack)

        radiomics_records.append(rad)

    # Save radiomics
    rad_df = pd.DataFrame(radiomics_records)
    rad_df.to_csv(f"{output_dir}/radiomics.csv", index=False)

    print(f"\n✓ Processed: {len(radiomics_records)} subjects")
    if failed:
        print(f"⚠ Failed: {len(failed)} — {failed[:5]}")
    print(f"✓ MRI arrays → {output_dir}/mri/")
    print(f"✓ Radiomics  → {output_dir}/radiomics.csv")

    # Merge radiomics back into participants df
    merged = df.merge(rad_df, on="subject_id", how="left")
    merged.to_csv(f"{output_dir}/participants_with_radiomics.csv", index=False)
    print(f"✓ Merged data → {output_dir}/participants_with_radiomics.csv")

    return merged, failed


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    merged_df, failed = run_mri_preprocessing(
        participants_csv="data/raw/synthetic/participants_validated.csv",
        data_dir="data/raw/synthetic",
        output_dir="data/processed",
        target_shape=(64, 64, 32),
        norm_method="z_score"
    )
    print("\nPreprocessing complete.")
    print("Next step: python src/preprocessing/clinical_preprocessor.py")
