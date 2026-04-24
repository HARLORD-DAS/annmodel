"""
==============================================================
  Phase 1 Master Runner
  Run this single script to complete all of Phase 1
==============================================================

Usage:
    python run_phase1.py --mode synthetic --n_subjects 200

What it does:
    Step 1: Generate synthetic dataset (or parse ISLES 2022)
    Step 2: MRI preprocessing (resample, normalize, radiomics)
    Step 3: Clinical preprocessing (impute, encode, scale, split)

After this completes, data/splits/ will have all training data
ready for Phase 2 (model building).
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_acquisition import generate_synthetic_dataset, check_data_integrity
from mri_preprocessor import run_mri_preprocessing
from clinical_preprocessor import run_clinical_preprocessing


def main():
    parser = argparse.ArgumentParser(description="Stroke ANN — Phase 1: Data Pipeline")
    parser.add_argument("--mode", choices=["synthetic", "isles2022"], default="synthetic")
    parser.add_argument("--n_subjects", type=int, default=200,
                        help="Number of synthetic subjects (ignored for isles2022)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  ANN Stroke Severity Prediction — Phase 1")
    print("  MSc Biotechnology Project")
    print("="*60)

    # ── Step 1: Dataset ──
    print("\n[Step 1/3] Dataset Acquisition")
    if args.mode == "synthetic":
        df = generate_synthetic_dataset(
            n_subjects=args.n_subjects,
            output_dir="data/raw/synthetic",
            seed=args.seed
        )
        check_data_integrity(df, data_dir="data/raw/synthetic")
        df.to_csv("data/raw/synthetic/participants_validated.csv", index=False)
        participants_csv = "data/raw/synthetic/participants_validated.csv"
        data_dir = "data/raw/synthetic"
    else:
        from src.preprocessing.dataset_acquisition import parse_isles2022
        df = parse_isles2022("data/raw/ISLES2022")
        df.to_csv("data/raw/isles2022_index.csv", index=False)
        participants_csv = "data/raw/isles2022_index.csv"
        data_dir = "data/raw/ISLES2022"

    # ── Step 2: MRI Preprocessing ──
    print("\n[Step 2/3] MRI Preprocessing")
    merged_df, failed = run_mri_preprocessing(
        participants_csv=participants_csv,
        data_dir=data_dir,
        output_dir="data/processed",
        target_shape=(64, 64, 32),
        norm_method="z_score"
    )

    # ── Step 3: Clinical Preprocessing ──
    print("\n[Step 3/3] Clinical + Radiomics Preprocessing")
    run_clinical_preprocessing(
        csv_path="data/processed/participants_with_radiomics.csv",
        splits_dir="data/splits",
        output_dir="outputs"
    )

    print("\n" + "="*60)
    print("  ✓ Phase 1 Complete!")
    print("  Data splits ready in: data/splits/")
    print("  Visualizations in:    outputs/")
    print("\n  Next: Run Phase 2 — ANN Model Training")
    print("  Command: python ann_model.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
