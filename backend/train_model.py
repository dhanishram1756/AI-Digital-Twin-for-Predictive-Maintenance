# train_model.py - Updated for Your CSV Format
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import joblib
from datetime import datetime
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

CSV_FILE = 'data/bearing_training_data.csv'
MODEL_OUTPUT = 'models/bearing_model_real.pkl'
METADATA_OUTPUT = 'models/model_metadata.pkl'
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================
# COLUMN MAPPING - IMPORTANT!
# Maps your CSV columns to model features
# ============================================

# Map your CSV columns to the required feature names
COLUMN_MAPPING = {
    # Your CSV column name -> Model feature name
    'Temperature': 'temperature',           # Your column: Temperature
    'Vibration_hz': 'vibration',            # Your column: Vibration_hz
    'RPM': 'rpm',                           # Your column: RPM
    'Barometer': 'pressure',                # Your column: Barometer (or A_Pres.PV)
    'Load_Percent': 'load',                 # Your column: Load_Percent
    'RUL_hours': 'rul',                     # Your column: RUL_hours
    # Optional additional features if available:
    # 'A_ACR_Mot.PV': 'current',            # Motor current
    # 'A_ACR_Pmp.PV': 'pump_current',       # Pump current
    # 'A_Temp.PV': 'temperature_alt',       # Alternative temperature
}

# Alternative mappings if your column names are slightly different
ALTERNATIVE_MAPPINGS = {
    'temperature': ['Temperature', 'Temp', 'A_Temp.PV'],
    'vibration': ['Vibration_hz', 'Vibration', 'Vib'],
    'rpm': ['RPM', 'Speed', 'RotorSpeed'],
    'pressure': ['Barometer', 'A_Pres.PV', 'Pressure', 'Pres'],
    'load': ['Load_Percent', 'Load', 'Load_%'],
    'rul': ['RUL_hours', 'RUL', 'RemainingLife', 'remaining_useful_life'],
    'current': ['A_ACR_Mot.PV', 'A_ACR_Pmp.PV', 'Current', 'A_ACR_Mot'],
    'speed': ['Speed', 'RPM', 'Velocity'],
    'maintenance_history': ['Maintenance', 'MaintDays', 'LastMaint']
}

# ============================================
# 1. LOAD AND MAP CSV DATA
# ============================================

def load_and_map_csv(csv_file):
    """Load CSV and map columns to model features"""
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file '{csv_file}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        return None, None, None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV: {csv_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Show first few rows to understand data
        print("\n📊 First 5 rows of data:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def map_columns_to_features(df):
    """Map CSV columns to model features using mappings"""
    
    feature_columns = {
        'temperature': None,
        'vibration': None,
        'rpm': None,
        'pressure': None,
        'load': None,
        'maintenance_history': None,
        'current': None,
        'speed': None,
        'rul': None  # Target variable
    }
    
    # First try direct mapping
    for csv_col, model_col in COLUMN_MAPPING.items():
        if csv_col in df.columns:
            if model_col in feature_columns:
                feature_columns[model_col] = csv_col
                print(f"   ✅ Mapped: {csv_col} -> {model_col}")
    
    # Then try alternative mappings for missing columns
    for model_col, alternatives in ALTERNATIVE_MAPPINGS.items():
        if feature_columns[model_col] is None:
            for alt in alternatives:
                if alt in df.columns:
                    feature_columns[model_col] = alt
                    print(f"   ✅ Mapped: {alt} -> {model_col}")
                    break
    
    # Check which columns were found
    found_features = {k: v for k, v in feature_columns.items() if v is not None}
    missing_features = {k: v for k, v in feature_columns.items() if v is None}
    
    print(f"\n📊 Column Mapping Results:")
    print(f"   Found: {len(found_features)}/{len(feature_columns)} columns")
    
    if missing_features:
        print(f"\n⚠️ Missing columns:")
        for col in missing_features:
            print(f"      {col}")
    
    # Check if RUL column was found
    if feature_columns['rul'] is None:
        print("\n❌ Error: No RUL column found!")
        print("   Please ensure your CSV has a column named:")
        print("   - RUL_hours (your column name)")
        print("   - Or one of: rul, RUL, remaining_useful_life, remaining_life")
        print(f"\n   Your columns: {df.columns.tolist()}")
        return None, None, None
    
    # Create feature DataFrame
    X = pd.DataFrame()
    y = None
    
    for model_col, csv_col in feature_columns.items():
        if csv_col is not None and model_col != 'rul':
            X[model_col] = df[csv_col]
        elif model_col == 'rul' and csv_col is not None:
            y = df[csv_col]
    
    # If we have less than 3 features, something is wrong
    if X.shape[1] < 3:
        print(f"\n❌ Error: Only {X.shape[1]} features found. Need at least 3.")
        print("   Please check your CSV columns and mapping.")
        return None, None, None
    
    # Fill missing maintenance_history if not available
    if 'maintenance_history' not in X.columns:
        print("   ⚠️ maintenance_history not found - using default value 30")
        X['maintenance_history'] = 30
    
    # Fill missing current if not available
    if 'current' not in X.columns:
        # Try to derive from load if available
        if 'load' in X.columns:
            X['current'] = 5 + (X['load'] / 100) * 15
            print("   ℹ️ Derived 'current' from 'load'")
        else:
            X['current'] = 10.5
            print("   ⚠️ current not found - using default value 10.5")
    
    # Fill missing speed if not available
    if 'speed' not in X.columns:
        if 'rpm' in X.columns:
            X['speed'] = X['rpm']
            print("   ℹ️ Using 'rpm' as 'speed'")
        else:
            X['speed'] = 1500
            print("   ⚠️ speed not found - using default value 1500")
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Ensure RUL is in days (if hours, convert to days)
    if y is not None:
        # Check if RUL is in hours (large numbers)
        if y.max() > 1000:
            print("   ℹ️ Converting RUL from hours to days (divide by 24)")
            y = y / 24
        
        # Clip RUL to reasonable range
        y = np.clip(y, 0, 365)
    
    print(f"\n📊 Final Data Summary:")
    print(f"   Features: {X.columns.tolist()}")
    print(f"   Samples: {len(X)}")
    print(f"   RUL range: {y.min():.1f} - {y.max():.1f} days")
    
    return X, y, X.columns.tolist()

# ============================================
# 2. TRAIN MODEL
# ============================================

def train_model(X, y, feature_names):
    """Train the RUL prediction model"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        ))
    ])
    
    print("\n🚀 Training Random Forest Regressor...")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print(f"\n📊 Model Performance:")
    print(f"   Training MAE: {train_mae:.2f} days")
    print(f"   Test MAE:     {test_mae:.2f} days")
    print(f"   Training RMSE: {train_rmse:.2f} days")
    print(f"   Test RMSE:     {test_rmse:.2f} days")
    print(f"   Training R²:   {train_r2:.4f}")
    print(f"   Test R²:       {test_r2:.4f}")
    print(f"   CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importance = model.named_steps['regressor'].feature_importances_
        print(f"\n📊 Feature Importance:")
        for feat, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
            print(f"   {feat:20s}: {imp:.4f}")
    
    # Test with sample inputs
    print(f"\n🧪 Sample Predictions:")
    sample_inputs = []
    
    # Create sample from actual data if available
    if len(X) > 0:
        # Use average values for samples
        avg = X.mean()
        std = X.std()
        
        # Healthy sample: lower than average vibration, temp
        healthy = avg.copy()
        healthy['vibration'] = avg['vibration'] * 0.5
        healthy['temperature'] = avg['temperature'] * 0.85
        
        # Warning sample: slightly elevated
        warning = avg.copy()
        warning['vibration'] = avg['vibration'] * 1.5
        warning['temperature'] = avg['temperature'] * 1.1
        
        # Critical sample: high values
        critical = avg.copy()
        critical['vibration'] = avg['vibration'] * 2.5
        critical['temperature'] = avg['temperature'] * 1.2
        critical['load'] = avg['load'] * 1.3
        
        sample_inputs = [
            (healthy, 'Healthy'),
            (warning, 'Warning'),
            (critical, 'Critical')
        ]
    
    for sample, label in sample_inputs:
        features_array = np.array([sample.values])
        rul = model.predict(features_array)[0]
        if rul > 100:
            status = 'Healthy'
        elif rul > 30:
            status = 'Warning'
        else:
            status = 'Critical'
        print(f"   {label}: RUL = {rul:.1f} days ({status})")
    
    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# ============================================
# 3. SAVE MODEL
# ============================================

def save_model(model, feature_names, metrics, X, y):
    """Save the trained model and metadata"""
    
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {MODEL_OUTPUT}")
    
    metadata = {
        'model_type': type(model.named_steps['regressor']).__name__,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'data_info': {
            'total_samples': len(X),
            'target_range': [y.min(), y.max()],
            'target_mean': y.mean(),
            'target_std': y.std()
        },
        'column_mapping': COLUMN_MAPPING
    }
    
    with open(METADATA_OUTPUT, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"💾 Metadata saved to: {METADATA_OUTPUT}")
    
    return metadata

# ============================================
# 4. GENERATE SYNTHETIC DATA (Fallback)
# ============================================

def generate_synthetic_data():
    """Generate synthetic data if CSV not found"""
    print("\n📊 Generating synthetic data...")
    
    np.random.seed(42)
    n_samples = 10000
    
    temperature = np.random.uniform(40, 90, n_samples)
    vibration = np.random.uniform(0.1, 3.0, n_samples)
    rpm = np.random.uniform(1000, 3000, n_samples)
    pressure = np.random.uniform(1, 10, n_samples)
    load = np.random.uniform(10, 100, n_samples)
    maintenance_history = np.random.uniform(0, 365, n_samples)
    current = np.random.uniform(5, 25, n_samples)
    speed = np.random.uniform(500, 3500, n_samples)
    
    degradation = (
        (temperature - 40) / 50 * 0.3 +
        (vibration - 0.5) / 2.5 * 0.4 +
        (load - 20) / 80 * 0.2 +
        (current - 5) / 20 * 0.1
    )
    degradation = np.clip(degradation, 0, 1)
    
    base_rul = np.random.uniform(50, 365, n_samples)
    rul = base_rul * (1 - degradation * 0.8)
    rul = np.clip(rul + np.random.normal(0, 10, n_samples), 1, 365)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'vibration': vibration,
        'rpm': rpm,
        'pressure': pressure,
        'load': load,
        'maintenance_history': maintenance_history,
        'current': current,
        'speed': speed,
        'rul': rul
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/bearing_training_data.csv', index=False)
    print(f"✅ Generated {len(df)} synthetic samples")
    print(f"   Saved to: data/bearing_training_data.csv")
    
    return df

# ============================================
# MAIN
# ============================================

def main():
    print("="*70)
    print("🔧 BEARING RUL MODEL TRAINER (CSV Input)")
    print("="*70)
    
    if not os.path.exists(CSV_FILE):
        print(f"\n⚠️ CSV file '{CSV_FILE}' not found.")
        response = input("Do you want to generate synthetic data? (y/n): ")
        if response.lower() == 'y':
            df = generate_synthetic_data()
        else:
            print("Please place your CSV file in the data/ directory and run again.")
            print(f"Expected file: {CSV_FILE}")
            sys.exit(1)
    else:
        df = load_and_map_csv(CSV_FILE)
        if df is None:
            print("\n❌ Failed to load data. Exiting.")
            return
    
    X, y, feature_names = map_columns_to_features(df)
    if X is None:
        print("\n❌ Failed to map columns. Exiting.")
        return
    
    model, metrics = train_model(X, y, feature_names)
    metadata = save_model(model, feature_names, metrics, X, y)
    
    print("\n" + "="*70)
    print("✅ Training complete! Model is ready for deployment.")
    print(f"   Model file: {MODEL_OUTPUT}")
    print(f"   Metadata: {METADATA_OUTPUT}")
    print(f"   Test R²: {metrics['test_r2']:.4f}")
    print(f"   Test MAE: {metrics['test_mae']:.2f} days")
    print("="*70)

if __name__ == '__main__':
    main()