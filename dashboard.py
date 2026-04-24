#!/usr/bin/env python3
"""
Streamlit Dashboard for Stroke Severity Prediction
Interactive web interface for model predictions and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stroke_predictor_service import StrokePredictorService
import json
import time

# Set page configuration
st.set_page_config(
    page_title="Stroke Severity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor service
@st.cache_resource
def load_predictor():
    """Load the prediction service (cached to avoid reloading)."""
    try:
        return StrokePredictorService()
    except Exception as e:
        st.error(f"Failed to load prediction service: {e}")
        return None

predictor = load_predictor()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .severity-mild {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .severity-moderate {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .severity-severe {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function."""

    # Header
    st.markdown('<h1 class="main-header">🧠 Stroke Severity Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**ANN-Based Multi-Modal Ischemic Stroke Severity & Progression Prediction**")

    if not predictor:
        st.error("❌ Prediction service failed to load. Please check the model files.")
        return

    # Sidebar
    st.sidebar.title("🔧 Controls")
    page = st.sidebar.radio("Navigation", ["Single Prediction", "Batch Prediction", "Model Info", "About"])

    if page == "Single Prediction":
        show_single_prediction()
    elif page == "Batch Prediction":
        show_batch_prediction()
    elif page == "Model Info":
        show_model_info()
    elif page == "About":
        show_about()

def show_single_prediction():
    """Single patient prediction interface."""
    st.header("🏥 Single Patient Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Patient Information")

        # Create tabs for different feature categories
        tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Clinical Assessment", "Medical History", "MRI Features"])

        patient_data = {}

        with tab1:
            st.subheader("📊 Demographics")
            col_a, col_b = st.columns(2)
            with col_a:
                patient_data['age'] = st.number_input("Age", min_value=0, max_value=120, value=65)
            with col_b:
                patient_data['gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

        with tab2:
            st.subheader("🏥 Clinical Assessment")
            col_a, col_b = st.columns(2)
            with col_a:
                patient_data['nihss_baseline'] = st.number_input("NIHSS Baseline Score", min_value=0, max_value=42, value=12)
            with col_b:
                patient_data['onset_to_door_hours'] = st.number_input("Onset to Door (hours)", min_value=0.0, max_value=72.0, value=2.5, step=0.5)

            patient_data['systolic_bp'] = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=160)
            patient_data['glucose_admission'] = st.number_input("Glucose Admission (mg/dL)", min_value=50, max_value=500, value=140)

        with tab3:
            st.subheader("📋 Medical History")
            col_a, col_b = st.columns(2)
            with col_a:
                patient_data['hypertension'] = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                patient_data['diabetes'] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            with col_b:
                patient_data['atrial_fibrillation'] = st.selectbox("Atrial Fibrillation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                patient_data['prior_stroke'] = st.selectbox("Prior Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with tab4:
            st.subheader("🧠 MRI Features")
            st.markdown("*Note: These values should come from MRI analysis*")

            col_a, col_b = st.columns(2)
            with col_a:
                patient_data['lesion_voxel_count'] = st.number_input("Lesion Voxel Count", min_value=0, value=15000, step=100)
                patient_data['dwi_mean_lesion'] = st.number_input("DWI Mean (lesion)", min_value=0.0, value=800.0, step=10.0)
                patient_data['dwi_std_lesion'] = st.number_input("DWI Std (lesion)", min_value=0.0, value=150.0, step=10.0)
                patient_data['adc_mean_lesion'] = st.number_input("ADC Mean (lesion)", min_value=0.0, value=600.0, step=10.0)
                patient_data['adc_std_lesion'] = st.number_input("ADC Std (lesion)", min_value=0.0, value=100.0, step=10.0)

            with col_b:
                patient_data['dwi_max_lesion'] = st.number_input("DWI Max (lesion)", min_value=0.0, value=1200.0, step=10.0)
                patient_data['dwi_min_lesion'] = st.number_input("DWI Min (lesion)", min_value=0.0, value=400.0, step=10.0)
                patient_data['adc_max_lesion'] = st.number_input("ADC Max (lesion)", min_value=0.0, value=900.0, step=10.0)
                patient_data['adc_min_lesion'] = st.number_input("ADC Min (lesion)", min_value=0.0, value=300.0, step=10.0)
                patient_data['penumbra_ratio'] = st.number_input("Penumbra Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

            # Additional MRI features
            st.subheader("Lesion Characteristics")
            col_c, col_d = st.columns(2)
            with col_c:
                patient_data['lesion_laterality'] = st.selectbox("Lesion Laterality", [0, 1], format_func=lambda x: "Left" if x == 0 else "Right")
                patient_data['relative_lesion_volume'] = st.number_input("Relative Lesion Volume", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
                patient_data['lesion_com_z'] = st.number_input("Lesion Center of Mass Z", min_value=-50.0, max_value=50.0, value=10.0, step=1.0)

            with col_d:
                st.subheader("Lesion Location (Multi-select)")
                locations = ['basal_ganglia', 'cerebellum', 'frontal', 'occipital', 'parietal', 'temporal']
                selected_locations = st.multiselect("Affected Brain Regions", locations, default=['frontal'])

                # Set location features
                for loc in locations:
                    patient_data[f'lesion_location_{loc}'] = 1.0 if loc in selected_locations else 0.0

    with col2:
        st.subheader("🎯 Prediction Results")

        if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Validate input
                    is_valid, errors = predictor.validate_input(patient_data)
                    if not is_valid:
                        st.error("❌ Invalid input data:")
                        for error in errors:
                            st.error(f"• {error}")
                        return

                    # Make prediction
                    result = predictor.predict(patient_data)

                    # Display results
                    severity = result['severity']
                    progression = result['progression']

                    # Severity prediction with color coding
                    severity_class = severity['class']
                    confidence = severity['probabilities'][severity_class]

                    if severity_class == 'mild':
                        st.markdown(f'<div class="prediction-card severity-mild">', unsafe_allow_html=True)
                    elif severity_class == 'moderate':
                        st.markdown(f'<div class="prediction-card severity-moderate">', unsafe_allow_html=True)
                    else:  # severe
                        st.markdown(f'<div class="prediction-card severity-severe">', unsafe_allow_html=True)

                    st.markdown("### Severity Prediction")
                    st.markdown(f"**Class:** {severity_class.upper()}")
                    st.markdown(".3f")
                    st.markdown("### Probabilities:")

                    # Progress bars for probabilities
                    probs = severity['probabilities']
                    st.progress(probs['mild'], text=f"Mild: {probs['mild']:.3f}")
                    st.progress(probs['moderate'], text=f"Moderate: {probs['moderate']:.3f}")
                    st.progress(probs['severe'], text=f"Severe: {probs['severe']:.3f}")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Progression prediction
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown("### Progression Prediction")
                    st.markdown(".3f")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Confidence indicator
                    confidence_level = result['confidence']['severity_confidence']
                    if confidence_level > 0.8:
                        st.success(f"🟢 High Confidence Prediction ({confidence_level:.3f})")
                    elif confidence_level > 0.6:
                        st.warning(f"🟡 Medium Confidence Prediction ({confidence_level:.3f})")
                    else:
                        st.error(f"🔴 Low Confidence Prediction ({confidence_level:.3f})")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)

def show_batch_prediction():
    """Batch prediction interface."""
    st.header("📊 Batch Prediction")

    st.markdown("Upload a CSV file with multiple patients for batch prediction.")

    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            if st.button("🔮 Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    # Convert DataFrame to list of dicts
                    patients_data = df.to_dict('records')

                    # Validate all patients
                    invalid_patients = []
                    for i, patient in enumerate(patients_data):
                        is_valid, errors = predictor.validate_input(patient)
                        if not is_valid:
                            invalid_patients.append((i, errors))

                    if invalid_patients:
                        st.error("❌ Some patients have invalid data:")
                        for idx, errors in invalid_patients[:5]:  # Show first 5
                            st.error(f"Patient {idx}: {errors}")
                        if len(invalid_patients) > 5:
                            st.error(f"... and {len(invalid_patients) - 5} more")
                        return

                    # Make batch predictions
                    results = predictor.batch_predict(patients_data)

                    # Display results
                    results_df = pd.DataFrame([
                        {
                            'Patient_ID': i,
                            'Severity_Class': r['severity']['class'],
                            'Severity_Confidence': r['severity']['probabilities'][r['severity']['class']],
                            'Progression_Predicted': r['progression']['predicted_value']
                        }
                        for i, r in enumerate(results)
                    ])

                    st.success("✅ Batch prediction completed!")
                    st.dataframe(results_df)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="stroke_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def show_model_info():
    """Display model information and feature details."""
    st.header("📋 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Model Architecture")
        st.markdown("""
        - **Type:** Multi-Modal ANN (Clinical + MRI features)
        - **Inputs:** 29 clinical and imaging features
        - **Outputs:** Severity classification (3 classes) + Progression regression
        - **Architecture:** Clinical branch → Fusion → Dual outputs
        - **Training:** 10 epochs on synthetic dataset
        """)

        st.subheader("📊 Performance Metrics")
        st.markdown("""
        - **Severity Classification:** Multi-class (Mild/Moderate/Severe)
        - **Progression Regression:** Continuous prediction
        - **Evaluation:** Cross-validation with baseline comparison
        """)

    with col2:
        st.subheader("🔧 Required Features")
        feature_info = predictor.get_feature_importance()

        for category, features in feature_info['feature_categories'].items():
            with st.expander(f"{category.replace('_', ' ').title()}"):
                for feature in features:
                    st.write(f"• {feature}")

def show_about():
    """About page with project information."""
    st.header("ℹ️ About This Project")

    st.markdown("""
    ## ANN-Based Stroke Severity Prediction

    This is a machine learning project for predicting ischemic stroke severity and progression using clinical data and MRI imaging features.

    ### 🎯 Objectives
    - Predict stroke severity (Mild/Moderate/Severe) using NIHSS scores
    - Forecast disease progression using multi-modal features
    - Provide interpretable predictions for clinical decision support

    ### 🧠 Model Features
    - **Multi-modal inputs:** Clinical assessments + MRI imaging data
    - **Dual outputs:** Classification (severity) + Regression (progression)
    - **Clinical validation:** Based on established medical guidelines

    ### 📊 Dataset
    - Synthetic dataset generated from medical literature
    - 29 features including demographics, vital signs, and MRI metrics
    - Balanced representation of severity classes

    ### 🛠️ Technology Stack
    - **Deep Learning:** TensorFlow/Keras
    - **Data Processing:** NumPy, Pandas, Scikit-learn
    - **Deployment:** Flask API, Streamlit Dashboard
    - **Visualization:** Matplotlib, Seaborn

    ---
    *MSc Biotechnology/Bioinformatics Project*
    """)

if __name__ == "__main__":
    main()