import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Bearing RUL Prediction", layout="wide")

# Title
st.title("🔧 Bearing RUL Prediction System")
st.markdown("*AI-powered predictive maintenance for bearing health monitoring*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Check if model exists
    model_path = 'backend/models/bearing_model.pkl'
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ No model found")
        
        # Option to train model
        if st.button("🔄 Train Model Now", type="primary"):
            st.info("Training model... (this may take a moment)")
            
            try:
                # Generate sample data if no data exists
                data_path = 'backend/data/training_data.csv'
                
                if not os.path.exists(data_path):
                    st.info("Generating sample data...")
                    # Create synthetic data
                    np.random.seed(42)
                    n_samples = 1000
                    X = np.random.randn(n_samples, 6)
                    # Create RUL based on features (synthetic relationship)
                    y = 100 - (X[:, 0] * 20 + X[:, 1] * 15 + X[:, 2] * 10 + 
                              X[:, 3] * 5 + X[:, 4] * 5 + X[:, 5] * 3)
                    y = np.maximum(y, 0)  # RUL can't be negative
                    
                    # Save sample data
                    os.makedirs('backend/data', exist_ok=True)
                    df = pd.DataFrame(X, columns=['Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                                                   'Temperature', 'Pressure', 'Speed'])
                    df['RUL'] = y
                    df.to_csv(data_path, index=False)
                    st.success("✅ Sample data created")
                
                # Load data
                df = pd.read_csv(data_path)
                X = df.drop('RUL', axis=1)
                y = df['RUL']
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Save model
                os.makedirs('backend/models', exist_ok=True)
                joblib.dump(model, model_path)
                st.success("✅ Model trained and saved!")
                st.rerun()  # ← FIXED: Use st.rerun() instead of experimental_rerun()
                
            except Exception as e:
                st.error(f"❌ Error training model: {e}")

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = 'backend/models/bearing_model.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

model = load_model()

# Main app
if model is not None:
    st.success("✅ Model ready for predictions")
    
    # Create input columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Sensor Input")
        
        # Input features
        features = {}
        feature_names = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                        'Temperature', 'Pressure', 'Speed']
        
        cols = st.columns(3)
        for i, name in enumerate(feature_names):
            with cols[i % 3]:
                features[name] = st.number_input(
                    name, 
                    value=0.0, 
                    format="%.4f",
                    key=name,
                    help=f"Enter {name} reading"
                )
        
        # Load simulation
        st.subheader("⚙️ Load Simulation")
        load_factor = st.slider("Load Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        
        if load_factor != 1.0:
            st.info(f"Load factor {load_factor:.1f}x applied to all sensor values")
    
    with col2:
        st.subheader("📈 Prediction")
        
        if st.button("🚀 Predict RUL", type="primary", use_container_width=True):
            try:
                # Prepare features
                feature_values = np.array([features[name] for name in feature_names])
                
                # Apply load factor
                feature_values = feature_values * load_factor
                feature_values = feature_values.reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(feature_values)
                rul = float(prediction[0])
                rul = max(0, min(rul, 100))  # Clamp between 0-100 for visualization
                
                # Display metrics
                st.metric("Remaining Useful Life", f"{rul:.2f} hours")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rul,
                    title={'text': "RUL (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Status
                if rul > 70:
                    st.success("✅ Bearing condition: **Good**")
                    st.info("🟢 Normal operation - Continue monitoring")
                elif rul > 30:
                    st.warning("⚠️ Bearing condition: **Monitor**")
                    st.info("🟡 Increased vibration detected - Schedule maintenance soon")
                else:
                    st.error("❌ Bearing condition: **Critical**")
                    st.info("🔴 Immediate maintenance required!")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    # Display model info
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ Model Info")
        st.info(f"""
        **Status:** ✅ Active
        **Type:** Random Forest
        **Features:** {len(feature_names)}
        **Port:** Optimized
        """)
        
        # Sample predictions
        if st.button("📊 Show Sample Prediction"):
            sample = {
                'Vibration_X': 2.5,
                'Vibration_Y': 1.8,
                'Vibration_Z': 0.9,
                'Temperature': 75.0,
                'Pressure': 45.0,
                'Speed': 1500.0
            }
            for name in feature_names:
                features[name] = sample[name]
            st.success("Sample values loaded!")
            st.rerun()  # ← FIXED: Use st.rerun() instead of experimental_rerun()
    
else:
    # Show instructions if no model
    st.warning("""
    ### ⚠️ No trained model found
    
    **To get started:**
    1. Click "Train Model Now" in the sidebar
    2. Or upload your own training data
    3. Or add a pre-trained model to `backend/models/bearing_model.pkl`
    """)
    
    with st.expander("📖 How to add training data"):
        st.markdown("""
        **Training data should be a CSV file with:**
        - Columns: `Vibration_X`, `Vibration_Y`, `Vibration_Z`, `Temperature`, `Pressure`, `Speed`
        - Target column: `RUL` (Remaining Useful Life)
        - Location: `backend/data/training_data.csv`
        """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Random Forest | Deployed on Streamlit Cloud")
