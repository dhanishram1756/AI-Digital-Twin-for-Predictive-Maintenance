# test_model.py - Test the trained model
import pickle
import numpy as np
import os

def test_model():
    """Test the saved model with sample inputs"""
    
    model_path = 'models/bearing_model_real.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first: python train_model.py")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("="*60)
        print("🧪 TESTING BEARING RUL MODEL")
        print("="*60)
        
        # Get feature count
        n_features = 8
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        elif hasattr(model, 'feature_names_in_'):
            n_features = len(model.feature_names_in_)
        
        print(f"📊 Model expects {n_features} features")
        
        # Test samples matching dashboard
        test_samples = [
            {
                'name': 'Healthy Bearing',
                'features': [65.0, 0.8, 1450.0, 5.2, 75.0, 30.0, 10.5, 1500.0]
            },
            {
                'name': 'Warning Bearing',
                'features': [78.0, 2.1, 1800.0, 7.5, 85.0, 10.0, 18.0, 2000.0]
            },
            {
                'name': 'Critical Bearing',
                'features': [89.0, 3.8, 2200.0, 9.2, 95.0, 5.0, 24.0, 2500.0]
            }
        ]
        
        print("\n📊 Test Results:")
        print("-"*60)
        
        for sample in test_samples:
            features = sample['features'][:n_features]
            features_array = np.array([features])
            
            rul = model.predict(features_array)[0]
            
            if rul > 100:
                status = '✅ Healthy'
                color = '\033[92m'
            elif rul > 30:
                status = '⚠️ Warning'
                color = '\033[93m'
            else:
                status = '🚨 Critical'
                color = '\033[91m'
            
            print(f"\n{sample['name']}:")
            print(f"   RUL: {rul:.1f} days")
            print(f"   Status: {color}{status}\033[0m")
        
        print("\n" + "="*60)
        print("✅ Model test complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == '__main__':
    test_model()