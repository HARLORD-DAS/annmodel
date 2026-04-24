#!/usr/bin/env python3
"""
Quick Access Script for Stroke Severity Prediction Model
Provides immediate access to model predictions.
"""

from stroke_predictor_service import StrokePredictorService
import json

def main():
    """Quick prediction demo."""
    print("🧠 Stroke Severity Prediction Model")
    print("=" * 40)

    # Initialize service
    try:
        predictor = StrokePredictorService()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Example patient data
    sample_patient = {
        'age': 65.0,
        'gender': 1.0,  # 1=male, 0=female
        'nihss_baseline': 12.0,
        'onset_to_door_hours': 2.5,
        'hypertension': 1.0,
        'diabetes': 0.0,
        'atrial_fibrillation': 0.0,
        'prior_stroke': 0.0,
        'systolic_bp': 160.0,
        'glucose_admission': 140.0,
        'lesion_voxel_count': 15000.0,
        'dwi_mean_lesion': 800.0,
        'dwi_std_lesion': 150.0,
        'dwi_max_lesion': 1200.0,
        'dwi_min_lesion': 400.0,
        'adc_mean_lesion': 600.0,
        'adc_std_lesion': 100.0,
        'adc_min_lesion': 300.0,
        'adc_max_lesion': 900.0,
        'penumbra_ratio': 0.3,
        'lesion_laterality': 1.0,  # 1=right, 0=left
        'relative_lesion_volume': 0.02,
        'lesion_com_z': 10.0,
        'lesion_location_basal_ganglia': 1.0,
        'lesion_location_cerebellum': 0.0,
        'lesion_location_frontal': 1.0,
        'lesion_location_occipital': 0.0,
        'lesion_location_parietal': 0.0,
        'lesion_location_temporal': 0.0
    }

    print("📊 Sample Patient Data:")
    print(f"   Age: {sample_patient['age']}")
    print(f"   NIHSS: {sample_patient['nihss_baseline']}")
    print(f"   Hypertension: {'Yes' if sample_patient['hypertension'] else 'No'}")
    print(f"   Lesion Volume: {sample_patient['lesion_voxel_count']}")
    print()

    # Make prediction
    try:
        result = predictor.predict(sample_patient)

        print("🎯 Prediction Results:")
        print("-" * 30)

        # Severity
        severity = result['severity']
        print(f"Severity Class: {severity['class'].upper()}")
        print(".3f")

        print("\nProbabilities:")
        probs = severity['probabilities']
        for cls, prob in probs.items():
            print(".3f")

        # Progression
        progression = result['progression']
        print(".3f")

        # Confidence
        confidence = result['confidence']['severity_confidence']
        confidence_level = "🟢 High" if confidence > 0.8 else "🟡 Medium" if confidence > 0.6 else "🔴 Low"
        print(f"\nConfidence: {confidence_level} ({confidence:.3f})")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")

    print("\n" + "=" * 40)
    print("🚀 Deployment Options:")
    print("   Web Dashboard: python deploy.py --dashboard")
    print("   REST API: python deploy.py --api")
    print("   CLI Tool: python predict_cli.py --help")
    print("   Python API: from stroke_predictor_service import StrokePredictorService")

if __name__ == "__main__":
    main()