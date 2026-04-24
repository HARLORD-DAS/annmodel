# ANN-Based Multi-Modal Prediction of Ischemic Stroke Severity and Progression
### Using MRI Features and Clinical Parameters
**MSc Biotechnology / Bioinformatics Project**

---

## Project Overview

This project builds an **Artificial Neural Network (ANN)** that predicts:
1. **Stroke severity class** — Mild / Moderate / Severe (classification)
2. **Neurological progression** — NIHSS change at 90 days (regression)

Using two input streams:
- **MRI features** — DWI, ADC volumes → radiomics extraction
- **Clinical parameters** — NIHSS, age, comorbidities, vitals

---
## 🚀 Model Deployment & Access

The trained ANN model can be accessed through multiple deployment options:

### Web Dashboard (Recommended)
```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Run interactive web dashboard
python deploy.py --dashboard
# or
streamlit run dashboard.py
```
- 🌐 **URL:** http://localhost:8501
- 📱 **Features:** Interactive forms, batch upload, real-time predictions

### REST API
```bash
# Run REST API server
python deploy.py --api
# or
python api.py
```
- 🔌 **URL:** http://localhost:5000
- 📡 **Endpoints:** `/predict`, `/batch_predict`, `/health`, `/features`
- 🛠️ **Integration:** Perfect for programmatic access

### Command Line Interface
```bash
# Single prediction
python predict_cli.py --age 65 --nihss 12 --hypertension 1

# Batch prediction from CSV
python predict_cli.py --csv patients.csv --output results.csv

# Interactive mode
python predict_cli.py --interactive
```

### Python API
```python
from stroke_predictor_service import StrokePredictorService

# Initialize service
predictor = StrokePredictorService()

# Make prediction
patient_data = {
    'age': 65, 'nihss_baseline': 12, 'hypertension': 1,
    # ... other features
}
result = predictor.predict(patient_data)
print(f"Severity: {result['severity']['class']}")
```

### Quick Test
```bash
python quick_predict.py
```

### Advanced Usage
```bash
# Install all dependencies
pip install -r requirements.txt

# Run deployment options
python deploy.py --help
```
## Project Structure

```
stroke_ann/
├── config.yaml                     ← Central configuration (all hyperparameters)
├── requirements.txt                ← All Python dependencies
├── run_phase1.py                   ← Phase 1 master runner
│
├── data/
│   ├── raw/
│   │   ├── synthetic/              ← Generated synthetic MRI + clinical data
│   │   └── ISLES2022/             ← Real dataset (download separately)
│   ├── processed/
│   │   ├── mri/                   ← Resampled + normalized .npy MRI arrays
│   │   ├── radiomics.csv          ← Extracted radiomics features
│   │   └── participants_with_radiomics.csv
│   └── splits/
│       ├── X_train/val/test.npy   ← Scaled feature arrays
│       ├── y_sev_*.npy            ← Severity labels (0/1/2)
│       ├── y_prog_*.npy           ← Progression values (NIHSS change)
│       ├── scaler.pkl             ← Fitted StandardScaler
│       ├── imputer.pkl            ← Fitted imputer
│       └── metadata.json          ← Feature names + split sizes
│
├── src/
│   ├── preprocessing/
│   │   ├── dataset_acquisition.py  ← Step 1: Download/generate data
│   │   ├── mri_preprocessor.py     ← Step 2: MRI pipeline
│   │   └── clinical_preprocessor.py ← Step 3: Tabular pipeline
│   ├── models/
│   │   └── ann_model.py            ← Phase 2: ANN architecture (coming next)
│   ├── evaluation/
│   │   └── evaluator.py            ← Phase 3: Metrics + benchmarking
│   └── visualization/
│       └── plots.py                ← Phase 4: All result plots
│   └── deployment/
│       ├── stroke_predictor_service.py ← Core prediction service
│       ├── api.py                  ← Flask REST API
│       ├── dashboard.py            ← Streamlit web dashboard
│       ├── predict_cli.py          ← Command-line interface
│       └── deploy.py               ← Deployment runner
│
├── notebooks/
│   └── 01_eda.ipynb               ← Exploratory data analysis notebook
├── outputs/                        ← Saved plots, metrics
├── results/                        ← Evaluation results + reports
├── checkpoints/                    ← Saved model weights
└── logs/                           ← Training logs (TensorBoard)
```

---

## Setup Instructions

### 1. Create Python environment
```bash
conda create -n stroke_ann python=3.10
conda activate stroke_ann
pip install -r requirements.txt
```

### 2. Run Phase 1 (Development mode — synthetic data)
```bash
python run_phase1.py --mode synthetic --n_subjects 200
```

### 3. Run Phase 1 (Real ISLES 2022 data)
1. Register at: https://isles22.grand-challenge.org/
2. Download dataset to `data/raw/ISLES2022/`
3. Run:
```bash
python run_phase1.py --mode isles2022
```

---

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Dataset & Preprocessing | ✅ Complete |
| **Phase 2** | ANN Model Design & Training | ✅ Complete |
| **Phase 3** | Evaluation & Interpretability | ✅ Complete |
| **Phase 4** | Hyperparameter Optimization | 🔜 |
| **Phase 5** | Baseline Benchmarking | 🔜 |
| **Phase 6** | Streamlit Dashboard | 🔜 |
| **Phase 7** | Report & Documentation | 🔜 |

---

## Dataset: ISLES 2022

- **Source**: https://isles22.grand-challenge.org/
- **Subjects**: 400 multi-center ischemic stroke MRI cases
- **Modalities**: DWI, ADC maps + lesion segmentation masks
- **License**: CC BY 4.0 (free for academic use)

**MRI modalities used:**
| Modality | Full Name | Clinical Relevance |
|----------|-----------|-------------------|
| DWI | Diffusion Weighted Imaging | Acute infarct core — hyperintense signal |
| ADC | Apparent Diffusion Coefficient | Cytotoxic edema — hypointense in ischemia |
| FLAIR | Fluid Attenuated Inversion Recovery | Penumbra & old lesions |

---

## Clinical Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Continuous | Patient age (years) |
| gender | Binary | 0=Female, 1=Male |
| nihss_baseline | Continuous | NIH Stroke Scale at admission (0–42) |
| onset_to_door_hours | Continuous | Symptom onset to hospital arrival |
| hypertension | Binary | History of hypertension |
| diabetes | Binary | History of diabetes mellitus |
| atrial_fibrillation | Binary | Cardiac arrhythmia (major stroke risk) |
| prior_stroke | Binary | Previous stroke history |
| systolic_bp | Continuous | Systolic blood pressure (mmHg) |
| glucose_admission | Continuous | Blood glucose at admission (mmol/L) |

**Severity classification (NIHSS-based):**
- Mild: NIHSS 1–6
- Moderate: NIHSS 7–15
- Severe: NIHSS 16–42

---

## Key References

1. Hernandez Petzsche et al. (2022). *ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset*. Scientific Data.
2. Kaka et al. (2021). *Artificial intelligence and deep learning in neuroradiology*. Insights into Imaging.
3. Stinear et al. (2019). *PREP2: A biomarker-based algorithm for predicting upper limb function after stroke*. Annals of Clinical & Translational Neurology.

---

## Author
MSc Biotechnology — Computational Biology & AI track
