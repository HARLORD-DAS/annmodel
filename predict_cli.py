#!/usr/bin/env python3
"""
Command-Line Interface for Stroke Severity Prediction
Allows batch predictions from CSV files and single predictions from command line.
"""

import argparse
import json
import pandas as pd
from stroke_predictor_service import StrokePredictorService
import sys

def load_predictor():
    """Load the prediction service."""
    try:
        return StrokePredictorService()
    except Exception as e:
        print(f"❌ Failed to load prediction service: {e}")
        sys.exit(1)

def predict_from_csv(csv_file, output_file=None):
    """Make predictions from CSV file."""
    predictor = load_predictor()

    try:
        # Load CSV data
        df = pd.read_csv(csv_file)
        print(f"📁 Loaded {len(df)} patients from {csv_file}")

        # Convert to list of dictionaries
        patients_data = df.to_dict('records')

        # Validate all patients
        print("🔍 Validating input data...")
        invalid_patients = []
        for i, patient in enumerate(patients_data):
            is_valid, errors = predictor.validate_input(patient)
            if not is_valid:
                invalid_patients.append((i, errors))

        if invalid_patients:
            print("❌ Validation errors found:")
            for idx, errors in invalid_patients[:5]:  # Show first 5
                print(f"  Patient {idx}: {errors}")
            if len(invalid_patients) > 5:
                print(f"  ... and {len(invalid_patients) - 5} more errors")
            sys.exit(1)

        # Make predictions
        print("🔮 Making predictions...")
        results = predictor.batch_predict(patients_data)

        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'patient_id': i,
                'severity_class': r['severity']['class'],
                'severity_confidence': r['severity']['probabilities'][r['severity']['class']],
                'progression_predicted': r['progression']['predicted_value'],
                'mild_prob': r['severity']['probabilities']['mild'],
                'moderate_prob': r['severity']['probabilities']['moderate'],
                'severe_prob': r['severity']['probabilities']['severe']
            }
            for i, r in enumerate(results)
        ])

        # Save or display results
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"✅ Results saved to {output_file}")
        else:
            print("\n📊 Prediction Results:")
            print(results_df.to_string(index=False))

        # Summary statistics
        print(f"\n📈 Summary:")
        print(f"   Total patients: {len(results_df)}")
        severity_counts = results_df['severity_class'].value_counts()
        for severity, count in severity_counts.items():
            print(f"   {severity.capitalize()}: {count} patients ({count/len(results_df)*100:.1f}%)")

    except Exception as e:
        print(f"❌ Error processing CSV: {str(e)}")
        sys.exit(1)

def predict_single(patient_data):
    """Make prediction for a single patient from command line args."""
    predictor = load_predictor()

    try:
        # Validate input
        is_valid, errors = predictor.validate_input(patient_data)
        if not is_valid:
            print("❌ Invalid input data:")
            for error in errors:
                print(f"   {error}")
            sys.exit(1)

        # Make prediction
        result = predictor.predict(patient_data)

        # Display results
        print("🎯 Prediction Results:")
        print("=" * 50)

        severity = result['severity']
        progression = result['progression']

        print(f"Severity Class: {severity['class'].upper()}")
        print(".3f")
        print("\nSeverity Probabilities:")
        for cls, prob in severity['probabilities'].items():
            print(".3f")

        print(".3f")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        sys.exit(1)

def interactive_mode():
    """Interactive mode for single patient prediction."""
    predictor = load_predictor()

    print("🧠 Interactive Stroke Severity Predictor")
    print("=" * 50)

    # Get feature information
    feature_info = predictor.get_feature_importance()

    patient_data = {}

    print("Please enter patient information:")
    print("(Press Enter for default values)")

    # Demographics
    print("\n📊 Demographics:")
    patient_data['age'] = float(input("Age [65]: ") or 65)
    patient_data['gender'] = int(input("Gender (0=Female, 1=Male) [1]: ") or 1)

    # Clinical
    print("\n🏥 Clinical Assessment:")
    patient_data['nihss_baseline'] = float(input("NIHSS Baseline [12]: ") or 12)
    patient_data['onset_to_door_hours'] = float(input("Onset to Door (hours) [2.5]: ") or 2.5)
    patient_data['systolic_bp'] = float(input("Systolic BP [160]: ") or 160)
    patient_data['glucose_admission'] = float(input("Glucose Admission [140]: ") or 140)

    # Medical History
    print("\n📋 Medical History (0=No, 1=Yes):")
    patient_data['hypertension'] = int(input("Hypertension [1]: ") or 1)
    patient_data['diabetes'] = int(input("Diabetes [0]: ") or 0)
    patient_data['atrial_fibrillation'] = int(input("Atrial Fibrillation [0]: ") or 0)
    patient_data['prior_stroke'] = int(input("Prior Stroke [0]: ") or 0)

    # MRI Features (simplified for CLI)
    print("\n🧠 MRI Features:")
    print("Note: Using default values for MRI features. Use CSV mode for full control.")
    patient_data.update({
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
        'lesion_laterality': 1.0,
        'relative_lesion_volume': 0.02,
        'lesion_com_z': 10.0,
        'lesion_location_basal_ganglia': 1.0,
        'lesion_location_cerebellum': 0.0,
        'lesion_location_frontal': 1.0,
        'lesion_location_occipital': 0.0,
        'lesion_location_parietal': 0.0,
        'lesion_location_temporal': 0.0
    })

    # Make prediction
    predict_single(patient_data)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Stroke Severity Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python predict_cli.py --interactive

  # Predict from CSV
  python predict_cli.py --csv patients.csv --output results.csv

  # Single prediction with args
  python predict_cli.py --age 65 --nihss 12 --hypertension 1
        """
    )

    # Input methods
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for single patient input')
    parser.add_argument('--csv', type=str,
                       help='CSV file with patient data for batch prediction')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for batch predictions')

    # Single patient arguments
    parser.add_argument('--age', type=float, help='Patient age')
    parser.add_argument('--gender', type=int, choices=[0, 1], help='Gender (0=Female, 1=Male)')
    parser.add_argument('--nihss', type=float, dest='nihss_baseline', help='NIHSS baseline score')
    parser.add_argument('--onset-hours', type=float, dest='onset_to_door_hours', help='Onset to door hours')
    parser.add_argument('--sbp', type=float, dest='systolic_bp', help='Systolic blood pressure')
    parser.add_argument('--glucose', type=float, dest='glucose_admission', help='Glucose admission')
    parser.add_argument('--hypertension', type=int, choices=[0, 1], help='Hypertension (0=No, 1=Yes)')
    parser.add_argument('--diabetes', type=int, choices=[0, 1], help='Diabetes (0=No, 1=Yes)')
    parser.add_argument('--afib', type=int, choices=[0, 1], dest='atrial_fibrillation', help='Atrial fibrillation (0=No, 1=Yes)')
    parser.add_argument('--prior-stroke', type=int, choices=[0, 1], dest='prior_stroke', help='Prior stroke (0=No, 1=Yes)')

    args = parser.parse_args()

    # Determine mode
    if args.interactive:
        interactive_mode()
    elif args.csv:
        predict_from_csv(args.csv, args.output)
    elif any([args.age, args.gender, args.nihss_baseline, args.onset_to_door_hours,
              args.systolic_bp, args.glucose_admission, args.hypertension, args.diabetes,
              args.atrial_fibrillation, args.prior_stroke]):
        # Single prediction from args
        patient_data = {
            'age': args.age or 65,
            'gender': args.gender if args.gender is not None else 1,
            'nihss_baseline': args.nihss_baseline or 12,
            'onset_to_door_hours': args.onset_to_door_hours or 2.5,
            'systolic_bp': args.systolic_bp or 160,
            'glucose_admission': args.glucose_admission or 140,
            'hypertension': args.hypertension if args.hypertension is not None else 1,
            'diabetes': args.diabetes if args.diabetes is not None else 0,
            'atrial_fibrillation': args.atrial_fibrillation if args.atrial_fibrillation is not None else 0,
            'prior_stroke': args.prior_stroke if args.prior_stroke is not None else 0,
        }

        # Add default MRI features
        patient_data.update({
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
            'lesion_laterality': 1.0,
            'relative_lesion_volume': 0.02,
            'lesion_com_z': 10.0,
            'lesion_location_basal_ganglia': 1.0,
            'lesion_location_cerebellum': 0.0,
            'lesion_location_frontal': 1.0,
            'lesion_location_occipital': 0.0,
            'lesion_location_parietal': 0.0,
            'lesion_location_temporal': 0.0
        })

        predict_single(patient_data)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()