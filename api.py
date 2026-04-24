#!/usr/bin/env python3
"""
Flask API for Stroke Severity Prediction Model
Provides REST API endpoints for model predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from stroke_predictor_service import StrokePredictorService
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web access

# Initialize prediction service
try:
    predictor = StrokePredictorService()
    print("✅ Stroke Predictor API initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize predictor service: {e}")
    predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if predictor else 'unhealthy',
        'service': 'Stroke Severity Predictor API',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a single patient."""
    try:
        if not predictor:
            return jsonify({'error': 'Prediction service not available'}), 503

        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate input
        is_valid, errors = predictor.validate_input(data)
        if not is_valid:
            return jsonify({'error': 'Invalid input data', 'details': errors}), 400

        # Make prediction
        result = predictor.predict(data)

        return jsonify({
            'success': True,
            'prediction': result
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple patients."""
    try:
        if not predictor:
            return jsonify({'error': 'Prediction service not available'}), 503

        # Get JSON data from request
        data = request.get_json()

        if not data or 'patients' not in data:
            return jsonify({'error': 'No patient data provided. Use {"patients": [...]}'}), 400

        patients = data['patients']
        if not isinstance(patients, list):
            return jsonify({'error': 'Patients must be a list'}), 400

        # Validate all inputs
        for i, patient in enumerate(patients):
            is_valid, errors = predictor.validate_input(patient)
            if not is_valid:
                return jsonify({
                    'error': f'Invalid data for patient {i}',
                    'details': errors
                }), 400

        # Make batch predictions
        results = predictor.batch_predict(patients)

        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })

    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get information about required features."""
    try:
        if not predictor:
            return jsonify({'error': 'Prediction service not available'}), 503

        feature_info = predictor.get_feature_importance()

        return jsonify({
            'success': True,
            'features': feature_info
        })

    except Exception as e:
        return jsonify({'error': 'Failed to get feature information', 'details': str(e)}), 500

@app.route('/example', methods=['GET'])
def get_example():
    """Get an example patient data structure."""
    example_patient = {
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

    return jsonify({
        'success': True,
        'example_patient': example_patient,
        'note': 'This is synthetic example data. Replace with real patient values.'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Stroke Predictor API...")
    print("📡 API endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /features - Get required features")
    print("   GET  /example - Get example patient data")
    print("   POST /predict - Single prediction")
    print("   POST /batch_predict - Batch predictions")
    print("\n🔗 API will be available at: http://localhost:5000")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )