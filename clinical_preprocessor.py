"""
==============================================================
  Phase 1 — Step 3: Clinical Data Preprocessing
  Handles tabular clinical + radiomics features
==============================================================

Pipeline:
  1. Load merged CSV (clinical + radiomics)
  2. Handle missing values (median imputation)
  3. Encode categorical variables (lesion location → one-hot)
  4. Encode target labels (severity → integer)
  5. Feature scaling (StandardScaler)
  6. Correlation & collinearity check
  7. Train / Validation / Test split (stratified)
  8. Save all splits + scaler object
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import yaml
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────
CLINICAL_FEATURES = [
    "age",
    "gender",
    "nihss_baseline",
    "onset_to_door_hours",
    "hypertension",
    "diabetes",
    "atrial_fibrillation",
    "prior_stroke",
    "systolic_bp",
    "glucose_admission",
]

RADIOMICS_FEATURES = [
    "lesion_volume_ml",
    "lesion_voxel_count",
    "dwi_mean_lesion",
    "dwi_std_lesion",
    "dwi_max_lesion",
    "dwi_min_lesion",
    "adc_mean_lesion",
    "adc_std_lesion",
    "adc_min_lesion",
    "adc_max_lesion",
    "penumbra_ratio",
    "lesion_laterality",
    "relative_lesion_volume",
    "lesion_com_z",
]

CATEGORICAL_FEATURES = ["lesion_location"]

TARGET_SEVERITY = "severity_label"
TARGET_PROGRESSION = "nihss_day90"

SEVERITY_MAP = {"mild": 0, "moderate": 1, "severe": 2}


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} subjects, {df.shape[1]} columns")
    print(f"  Severity distribution:\n{df[TARGET_SEVERITY].value_counts()}\n")
    return df


# ──────────────────────────────────────────────
# 2. Missing value imputation
# ──────────────────────────────────────────────
def impute_missing(df, method="median", fit_scaler=None):
    """
    Impute missing numeric values.
    method: 'median', 'mean', 'knn'
    Returns: (df_imputed, imputer)
    """
    num_cols = CLINICAL_FEATURES + RADIOMICS_FEATURES
    available = [c for c in num_cols if c in df.columns]

    missing_report = df[available].isnull().sum()
    if missing_report.sum() > 0:
        print(f"\nMissing values detected:")
        print(missing_report[missing_report > 0])
    else:
        print("✓ No missing values in numeric features")

    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=method)

    df[available] = imputer.fit_transform(df[available])
    return df, imputer


# ──────────────────────────────────────────────
# 3. Encode categorical features
# ──────────────────────────────────────────────
def encode_categoricals(df):
    """
    One-hot encode lesion_location.
    Returns: (df_encoded, list of new column names)
    """
    encoded_cols = []

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)
        encoded_cols.extend(dummies.columns.tolist())
        print(f"✓ One-hot encoded '{col}' → {len(dummies.columns)} columns")

    return df, encoded_cols


# ──────────────────────────────────────────────
# 4. Encode target labels
# ──────────────────────────────────────────────
def encode_targets(df):
    """
    Map severity_label → integer (0/1/2)
    Compute: nihss_change = nihss_baseline - nihss_day90  (positive = improvement)
    """
    df["severity_int"] = df[TARGET_SEVERITY].map(SEVERITY_MAP)
    df["nihss_change"] = df["nihss_baseline"] - df[TARGET_PROGRESSION]
    print(f"✓ Severity encoded: {SEVERITY_MAP}")
    print(f"✓ NIHSS change (outcome) stats:")
    print(f"  Mean: {df['nihss_change'].mean():.2f}, Std: {df['nihss_change'].std():.2f}")
    return df


# ──────────────────────────────────────────────
# 5. Feature scaling
# ──────────────────────────────────────────────
def scale_features(X_train, X_val, X_test, method="standard"):
    """
    Fit scaler on training set only. Transform val and test.
    Prevents data leakage.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, scaler


# ──────────────────────────────────────────────
# 6. Correlation analysis
# ──────────────────────────────────────────────
def plot_correlation_matrix(df, feature_cols, output_dir="outputs"):
    """Save a correlation heatmap of all features."""
    os.makedirs(output_dir, exist_ok=True)
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr, annot=False, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation matrix saved → {output_dir}/correlation_matrix.png")

    # Flag high correlations (>0.85 may indicate redundancy)
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.85:
                high_corr.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))
    if high_corr:
        print(f"\n⚠ High correlation pairs (>0.85) — consider dropping one:")
        for a, b, r in high_corr:
            print(f"  {a} ↔ {b}: r={r}")

    return corr


# ──────────────────────────────────────────────
# 7. Mutual information feature importance
# ──────────────────────────────────────────────
def mutual_info_ranking(X, y, feature_names, output_dir="outputs"):
    """Rank features by mutual information with severity target."""
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": feature_names, "mi_score": mi})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    ax.barh(mi_df["feature"], mi_df["mi_score"], color="#2563EB")
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Feature Importance (Mutual Information vs Severity)", fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mutual_info_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Mutual info ranking saved → {output_dir}/mutual_info_ranking.png")
    print("\nTop 10 features by MI score:")
    print(mi_df.head(10).to_string(index=False))

    return mi_df


# ──────────────────────────────────────────────
# 8. Train / Val / Test split
# ──────────────────────────────────────────────
def stratified_split(df, feature_cols, train=0.70, val=0.15, test=0.15, seed=42):
    """
    Stratified split on severity_int to preserve class distribution.
    Returns: (X_train, X_val, X_test, y_sev_train, ..., y_prog_train, ..., indices)
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    X = df[feature_cols].values
    y_sev = df["severity_int"].values
    y_prog = df["nihss_change"].values
    subj_ids = df["subject_id"].values

    # First: split off test set
    X_temp, X_test, y_sev_temp, y_sev_test, y_prog_temp, y_prog_test, ids_temp, ids_test = \
        train_test_split(X, y_sev, y_prog, subj_ids,
                         test_size=test, stratify=y_sev, random_state=seed)

    # Then: split remainder into train and val
    val_relative = val / (train + val)
    X_train, X_val, y_sev_train, y_sev_val, y_prog_train, y_prog_val, ids_train, ids_val = \
        train_test_split(X_temp, y_sev_temp, y_prog_temp, ids_temp,
                         test_size=val_relative, stratify=y_sev_temp, random_state=seed)

    print(f"\n✓ Dataset split (stratified by severity):")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for split_name, y in [("Train", y_sev_train), ("Val", y_sev_val), ("Test", y_sev_test)]:
        uniq, counts = np.unique(y, return_counts=True)
        dist = {["mild","moderate","severe"][u]: int(c) for u, c in zip(uniq, counts)}
        print(f"  {split_name} class dist: {dist}")

    return (X_train, X_val, X_test,
            y_sev_train, y_sev_val, y_sev_test,
            y_prog_train, y_prog_val, y_prog_test,
            ids_train, ids_val, ids_test)


# ──────────────────────────────────────────────
# 9. Save all splits
# ──────────────────────────────────────────────
def save_splits(splits_dir, X_train, X_val, X_test,
                y_sev_train, y_sev_val, y_sev_test,
                y_prog_train, y_prog_val, y_prog_test,
                ids_train, ids_val, ids_test,
                feature_cols, scaler, imputer):
    """Save all split arrays + scaler + metadata."""
    os.makedirs(splits_dir, exist_ok=True)

    np.save(f"{splits_dir}/X_train.npy", X_train)
    np.save(f"{splits_dir}/X_val.npy", X_val)
    np.save(f"{splits_dir}/X_test.npy", X_test)

    np.save(f"{splits_dir}/y_sev_train.npy", y_sev_train)
    np.save(f"{splits_dir}/y_sev_val.npy", y_sev_val)
    np.save(f"{splits_dir}/y_sev_test.npy", y_sev_test)

    np.save(f"{splits_dir}/y_prog_train.npy", y_prog_train)
    np.save(f"{splits_dir}/y_prog_val.npy", y_prog_val)
    np.save(f"{splits_dir}/y_prog_test.npy", y_prog_test)

    np.save(f"{splits_dir}/ids_train.npy", ids_train)
    np.save(f"{splits_dir}/ids_val.npy", ids_val)
    np.save(f"{splits_dir}/ids_test.npy", ids_test)

    # Save scaler and imputer
    with open(f"{splits_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{splits_dir}/imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)

    # Save feature metadata
    meta = {
        "feature_columns": list(feature_cols),
        "n_features": len(feature_cols),
        "severity_map": SEVERITY_MAP,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    with open(f"{splits_dir}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ All splits saved → {splits_dir}/")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Files: X_train/val/test, y_sev, y_prog, ids, scaler, imputer, metadata")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def run_clinical_preprocessing(
    csv_path="data/processed/participants_with_radiomics.csv",
    splits_dir="data/splits",
    output_dir="outputs"
):
    print(f"\n{'='*55}")
    print(f"  Clinical + Radiomics Preprocessing Pipeline")
    print(f"{'='*55}\n")

    # 1. Load
    df = load_data(csv_path)

    # 2. Encode categoricals
    df, encoded_cols = encode_categoricals(df)

    # 3. Encode targets
    df = encode_targets(df)

    # 4. Impute
    df, imputer = impute_missing(df, method="median")

    # 5. Build feature set
    feature_cols = CLINICAL_FEATURES + RADIOMICS_FEATURES + encoded_cols
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\n✓ Total features selected: {len(feature_cols)}")

    # 6. Correlation analysis
    plot_correlation_matrix(df, feature_cols, output_dir)

    # 7. Stratified split
    (X_train, X_val, X_test,
     y_sev_train, y_sev_val, y_sev_test,
     y_prog_train, y_prog_val, y_prog_test,
     ids_train, ids_val, ids_test) = stratified_split(df, feature_cols)

    # 8. Scale (fit on train only)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # 9. Mutual info ranking (on scaled train set)
    mutual_info_ranking(X_train_s, y_sev_train, feature_cols, output_dir)

    # 10. Save everything
    save_splits(
        splits_dir,
        X_train_s, X_val_s, X_test_s,
        y_sev_train, y_sev_val, y_sev_test,
        y_prog_train, y_prog_val, y_prog_test,
        ids_train, ids_val, ids_test,
        feature_cols, scaler, imputer
    )

    print(f"\n{'='*55}")
    print(f"  Phase 1 COMPLETE")
    print(f"  Next step: python src/models/ann_model.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    run_clinical_preprocessing(
        csv_path="data/processed/participants_with_radiomics.csv",
        splits_dir="data/splits",
        output_dir="outputs"
    )
