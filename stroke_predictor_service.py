#!/usr/bin/env python3
"""
Stroke Severity Prediction Model Deployment Service
Provides easy access to the trained ANN model for predictions.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from typing import Dict, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

class StrokePredictorService:
    """Service class for stroke severity prediction model deployment."""

    def __init__(self, model_path='checkpoints/improved_model.keras',
                 scaler_path='data/splits/scaler.pkl',
                 imputer_path='data/splits/imputer.pkl',
                 metadata_path='data/splits/metadata.json'):
        """Initialize the prediction service."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.imputer_path = imputer_path
        self.metadata_path = metadata_path

        # Load model and preprocessing components
        self.load_model()
        self.load_preprocessing()
        self.load_metadata()

        print("✅ Stroke Predictor Service initialized successfully!")

    def load_model(self):
        """Load the trained ANN model."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def load_preprocessing(self):
        """Load preprocessing components (scaler, imputer)."""
        try:
            self.scaler = joblib.load(self.scaler_path)
            self.imputer = joblib.load(self.imputer_path)

            # Get imputer feature names
            if hasattr(self.imputer, 'feature_names_in_'):
                self.imputer_features = list(self.imputer.feature_names_in_)
            else:
                # Fallback: assume first N features where N is imputer input size
                self.imputer_features = self.feature_columns[:self.imputer.n_features_in_]

            print(f"✅ Preprocessing components loaded")
            print(f"   Imputer trained on {len(self.imputer_features)} features")
            print(f"   Scaler expects {self.scaler.n_features_in_} features")

        except Exception as e:
            raise Exception(f"Failed to load preprocessing components: {str(e)}")

    def load_metadata(self):
        """Load feature metadata."""
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata['feature_columns']
            self.severity_map = self.metadata['severity_map']
            self.severity_labels = {v: k for k, v in self.severity_map.items()}
            print(f"✅ Metadata loaded - {len(self.feature_columns)} features")
        except Exception as e:
            raise Exception(f"Failed to load metadata: {str(e)}")

    def preprocess_input(self, input_data: Dict[str, Union[float, int]]) -> np.ndarray:
        """Preprocess input data for model prediction.

        Args:
            input_data: Dictionary with feature names as keys and values as values

        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Reorder columns to match training data
        df = df[self.feature_columns]

        # Handle imputation for features that imputer knows about
        df_imputed = df.copy()
        imputer_df = df[self.imputer_features]
        imputed_values = self.imputer.transform(imputer_df)
        df_imputed[self.imputer_features] = imputed_values

        # For features not known to imputer, fill with 0 (they should be present)
        non_imputer_features = set(self.feature_columns) - set(self.imputer_features)
        for feature in non_imputer_features:
            if df[feature].isnull().any():
                df_imputed[feature] = df_imputed[feature].fillna(0)

        # Scale all features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_imputed),
            columns=self.feature_columns
        )

        return df_scaled.values

    def predict(self, input_data: Dict[str, Union[float, int]]) -> Dict[str, Union[str, float]]:
        """Make prediction for stroke severity and progression.

        Args:
            input_data: Dictionary with patient features

        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess input
        X = self.preprocess_input(input_data)

        # Make prediction
        predictions = self.model.predict(X, verbose=0)

        # Extract severity and progression predictions
        severity_probs = predictions[0][0]  # Shape: (1, 3) -> (3,)
        progression_pred = predictions[1][0][0]  # Shape: (1, 1) -> scalar

        # Get severity class
        severity_class_idx = np.argmax(severity_probs)
        severity_class = self.severity_labels[severity_class_idx]

        # Create result dictionary
        result = {
            'severity': {
                'class': severity_class,
                'class_index': int(severity_class_idx),
                'probabilities': {
                    'mild': float(severity_probs[0]),
                    'moderate': float(severity_probs[1]),
                    'severe': float(severity_probs[2])
                }
            },
            'progression': {
                'predicted_value': float(progression_pred),
                'unit': 'progression_score'  # You can specify the actual unit
            },
            'confidence': {
                'severity_confidence': float(np.max(severity_probs)),
                'prediction_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        return result

    def get_feature_importance(self) -> Dict[str, List[str]]:
        """Get information about model features."""
        return {
            'required_features': self.feature_columns,
            'n_features': len(self.feature_columns),
            'severity_classes': list(self.severity_map.keys()),
            'feature_categories': {
                'demographics': ['age', 'gender'],
                'clinical_assessment': ['nihss_baseline', 'onset_to_door_hours'],
                'medical_history': ['hypertension', 'diabetes', 'atrial_fibrillation', 'prior_stroke'],
                'vital_signs': ['systolic_bp', 'glucose_admission'],
                'mri_features': [col for col in self.feature_columns if 'lesion' in col or col.startswith(('dwi_', 'adc_'))]
            }
        }

    def batch_predict(self, input_data_list: List[Dict[str, Union[float, int]]]) -> List[Dict]:
        """Make batch predictions for multiple patients.

        Args:
            input_data_list: List of dictionaries with patient features

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(data) for data in input_data_list]

    def validate_input(self, input_data: Dict[str, Union[float, int]]) -> Tuple[bool, List[str]]:
        """Validate input data structure and values.

        Args:
            input_data: Input data dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required features
        missing_features = set(self.feature_columns) - set(input_data.keys())
        if missing_features:
            errors.append(f"Missing required features: {missing_features}")

        # Check data types
        for feature in input_data:
            if feature in self.feature_columns:
                value = input_data[feature]
                if not isinstance(value, (int, float)):
                    errors.append(f"Feature '{feature}' must be numeric, got {type(value)}")

        # Check for reasonable value ranges (basic validation)
        if 'age' in input_data:
            age = input_data['age']
            if not (0 <= age <= 120):
                errors.append(f"Age must be between 0-120, got {age}")

        if 'nihss_baseline' in input_data:
            nihss = input_data['nihss_baseline']
            if not (0 <= nihss <= 42):
                errors.append(f"NIHSS score must be between 0-42, got {nihss}")

        return len(errors) == 0, errors


# Example usage and testing
def example_usage():
    """Example of how to use the prediction service."""
    # Initialize service
    predictor = StrokePredictorService()

    # Example patient data (you would replace this with real data)
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

    # Make prediction
    result = predictor.predict(sample_patient)

    print("🎯 Prediction Result:")
    print(f"Severity: {result['severity']['class']} (confidence: {result['severity']['probabilities'][result['severity']['class']]:.3f})")
    print(f"Progression: {result['progression']['predicted_value']:.3f}")

    return result


if __name__ == "__main__":
    # Run example
    example_usage()