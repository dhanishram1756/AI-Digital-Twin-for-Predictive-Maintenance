# app.py - Complete Flask Backend for Bearing RUL Prediction
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import logging
import os
import traceback
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = 'models/bearing_model_real.pkl'
MODEL_METADATA_PATH = 'models/model_metadata.pkl'

# ============================================
# GLOBAL STATE
# ============================================

model = None
model_features = []
model_info = {
    'name': 'Bearing RUL Predictor',
    'version': '1.0',
    'type': 'Unknown',
    'n_features': 0,
    'is_loaded': False
}

# Default feature names (fallback if model doesn't provide them)
DEFAULT_FEATURES = [
    'temperature', 'vibration', 'rpm', 'pressure', 
    'load', 'maintenance_history', 'current', 'speed'
]

# ============================================
# MODEL LOADING
# ============================================

def load_model():
    """Load the pickle model and extract feature information"""
    global model, model_features, model_info
    
    model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        logger.error(f"   Current directory: {os.getcwd()}")
        logger.error(f"   Please ensure '{model_path}' exists in this directory")
        return False
    
    try:
        logger.info(f"📦 Loading model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)
        
        # Handle different model types
        if hasattr(loaded, 'named_steps'):
            # It's a Pipeline
            model = loaded
            model_info['type'] = 'Pipeline'
            model_info['is_pipeline'] = True
            
            # Get the actual estimator
            for name, step in loaded.named_steps.items():
                if hasattr(step, 'predict'):
                    model_info['estimator'] = name
                    if hasattr(step, 'feature_names_in_'):
                        model_features = list(step.feature_names_in_)
                        break
        else:
            model = loaded
            model_info['type'] = type(loaded).__name__
            model_info['is_pipeline'] = False
        
        # Get feature names
        if not model_features:
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
            elif hasattr(model, 'n_features_in_'):
                n = model.n_features_in_
                model_features = [f'feature_{i+1}' for i in range(n)]
            else:
                # Try to get from metadata
                if os.path.exists(MODEL_METADATA_PATH):
                    with open(MODEL_METADATA_PATH, 'rb') as f:
                        metadata = pickle.load(f)
                        if 'feature_names' in metadata:
                            model_features = metadata['feature_names']
        
        # If still no features, use defaults
        if not model_features:
            model_features = DEFAULT_FEATURES
        
        model_info['n_features'] = len(model_features)
        model_info['is_loaded'] = True
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Type: {model_info['type']}")
        logger.info(f"   Features: {model_info['n_features']}")
        logger.info(f"   Feature names: {model_features[:5]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

def load_metadata():
    """Load model metadata if available"""
    try:
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
                logger.info(f"📋 Loaded metadata: {metadata.get('training_date', 'Unknown')}")
                return metadata
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
    return None

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_rul(features):
    """Make RUL prediction using the loaded model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    try:
        # Convert to numpy array
        features_array = np.array([features]).astype(float)
        
        # Make prediction
        rul = model.predict(features_array)[0]
        
        # Ensure RUL is non-negative
        rul = max(0, float(rul))
        
        # Round to 1 decimal
        rul = round(rul, 1)
        
        # Determine status and message
        status, message, color = get_status_and_message(rul)
        
        # Calculate health score (0-100%)
        health_score = min(100, (rul / 365) * 100)
        
        return {
            'rul_days': rul,
            'status': status,
            'message': message,
            'color': color,
            'health_score': round(health_score, 1),
            'confidence': calculate_confidence(features, rul)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

def get_status_and_message(rul):
    """Determine status and recommendation based on RUL"""
    if rul > 200:
        return 'Healthy', 'Bearing is in excellent condition. Continue regular monitoring.', '#3B82F6'
    elif rul > 100:
        return 'Healthy', 'Bearing condition is good. No immediate action needed.', '#3B82F6'
    elif rul > 60:
        return 'Warning', 'Bearing showing early signs of wear. Schedule inspection within 60 days.', '#F59E0B'
    elif rul > 30:
        return 'Warning', 'Bearing degradation detected. Plan maintenance within 30 days.', '#F59E0B'
    elif rul > 10:
        return 'Critical', 'Bearing is in critical condition. Schedule replacement within 10 days.', '#DC2626'
    else:
        return 'Critical', 'Bearing failure imminent! Immediate replacement required!', '#DC2626'

def calculate_confidence(features, rul):
    """Calculate prediction confidence based on feature values"""
    # Define normal ranges
    normal_ranges = {
        'temperature': (40, 80),
        'vibration': (0.1, 2.0),
        'rpm': (1000, 2500),
        'pressure': (1, 8),
        'load': (20, 80),
        'maintenance_history': (0, 60),
        'current': (5, 20),
        'speed': (500, 3000)
    }
    
    confidence = 0.9
    
    for i, feature in enumerate(model_features):
        if i < len(features) and feature in normal_ranges:
            min_val, max_val = normal_ranges[feature]
            val = features[i]
            
            if val < min_val or val > max_val:
                deviation = min(abs(val - min_val), abs(val - max_val))
                if deviation > 10:
                    confidence -= 0.05
    
    if rul > 300:
        confidence *= 0.85
    elif rul < 5:
        confidence *= 0.80
    
    return max(0.5, min(0.95, confidence))

def map_sensor_data(data):
    """Map incoming sensor data to model features"""
    features = []
    
    if 'features' in data:
        features = data['features']
    else:
        for f in model_features:
            if f in data:
                features.append(float(data[f]))
            else:
                found = False
                for key, value in data.items():
                    if key.lower() == f.lower() or key.lower().replace('_', '') == f.lower().replace('_', ''):
                        features.append(float(value))
                        found = True
                        break
                if not found:
                    logger.warning(f"Missing feature: {f}, using 0")
                    features.append(0.0)
    
    return features

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the dashboard"""
    try:
        return render_template('dashboard.html', 
                              features=model_features,
                              model_info=model_info)
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Dashboard template not found. Please ensure templates/dashboard.html exists.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_info['is_loaded'] else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'model_type': model_info.get('type', 'Unknown'),
        'features': model_features,
        'n_features': model_info.get('n_features', 0)
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get model feature information"""
    return jsonify({
        'features': model_features,
        'count': len(model_features),
        'model_type': model_info.get('type', 'Unknown'),
        'is_loaded': model_info.get('is_loaded', False)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please check server logs.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        logger.info(f"📊 Prediction request: {data}")
        
        features = map_sensor_data(data)
        
        if len(features) != model_info['n_features']:
            return jsonify({
                'status': 'error',
                'message': f'Expected {model_info["n_features"]} features, got {len(features)}'
            }), 400
        
        result = predict_rul(features)
        result['timestamp'] = datetime.now().isoformat()
        result['features'] = {f: v for f, v in zip(model_features, features)}
        
        logger.info(f"✅ Prediction: RUL={result['rul_days']:.1f}, Status={result['status']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Simulate the effect of load increase on RUL"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please check server logs.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        load_increase = float(data.get('load_increase', 0.2))
        load_increase = max(0, min(1, load_increase))
        
        logger.info(f"📊 Simulation request: +{load_increase*100:.0f}% load")
        
        features = map_sensor_data(data)
        
        if len(features) != model_info['n_features']:
            return jsonify({
                'status': 'error',
                'message': f'Expected {model_info["n_features"]} features, got {len(features)}'
            }), 400
        
        sim_features = features.copy()
        for i, f in enumerate(model_features):
            lower_f = f.lower()
            if any(key in lower_f for key in ['load', 'current', 'rpm', 'speed']):
                sim_features[i] = features[i] * (1 + load_increase * 0.5)
        
        result = predict_rul(sim_features)
        result['isSimulation'] = True
        result['load_increase'] = load_increase
        result['timestamp'] = datetime.now().isoformat()
        result['original_rul'] = predict_rul(features)['rul_days']
        
        logger.info(f"✅ Simulation: RUL={result['rul_days']:.1f} (original: {result['original_rul']:.1f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Simulation error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple samples"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please check server logs.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No samples provided'
            }), 400
        
        samples = data['samples']
        results = []
        
        for i, sample in enumerate(samples):
            try:
                features = map_sensor_data(sample)
                if len(features) == model_info['n_features']:
                    result = predict_rul(features)
                    result['sample_id'] = i
                    results.append(result)
                else:
                    results.append({
                        'sample_id': i,
                        'status': 'error',
                        'message': f'Expected {model_info["n_features"]} features, got {len(features)}'
                    })
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'status': 'error',
                    'message': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': [
            '/', '/api/health', '/api/features', 
            '/api/predict', '/api/simulate', '/api/batch_predict'
        ]
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 BEARING RUL PREDICTION SYSTEM")
    print("="*70)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📄 Model file: {MODEL_PATH}")
    print("="*70)
    
    if load_model():
        print("\n✅ Model loaded successfully!")
        print(f"   Type: {model_info['type']}")
        print(f"   Features: {model_info['n_features']}")
        print(f"   Feature names: {model_features}")
    else:
        print("\n⚠️ Model not loaded. Running in demo mode.")
        model_features = DEFAULT_FEATURES
        model_info['n_features'] = len(DEFAULT_FEATURES)
        model_info['is_loaded'] = False
    
    metadata = load_metadata()
    if metadata:
        print(f"\n📋 Model metadata:")
        print(f"   Training date: {metadata.get('training_date', 'Unknown')}")
        if 'metrics' in metadata:
            print(f"   Test R²: {metadata['metrics'].get('test_r2', 'Unknown')}")
    
    print("\n" + "="*70)
    print("🌐 Server Information:")
    print("   URL: http://localhost:8000")
    print("   Dashboard: http://localhost:8000/")
    print("   API Health: http://localhost:8000/api/health")
    print("   API Features: http://localhost:8000/api/features")
    print("="*70)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=8000)
