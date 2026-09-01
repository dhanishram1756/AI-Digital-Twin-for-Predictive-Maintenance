import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Bearing RUL Prediction", layout="wide")

st.title("🔧 Bearing RUL Prediction System")
st.markdown("*AI-powered predictive maintenance for bearing health monitoring*")

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = 'backend/models/bearing_model.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except:
        pass
    return None

model = load_model()

# Create input fields
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Sensor Input")
    features = {}
    feature_names = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                     'Temperature', 'Pressure', 'Speed']
    for name in feature_names:
        features[name] = st.number_input(name, value=0.0, format="%.4f")

with col2:
    st.subheader("📈 Prediction")
    if st.button("🚀 Predict RUL", type="primary"):
        if model:
            try:
                feature_values = np.array(list(features.values())).reshape(1, -1)
                prediction = model.predict(feature_values)
                rul = float(prediction[0])
                st.metric("Remaining Useful Life", f"{rul:.2f} hours")
                if rul > 70:
                    st.success("✅ Condition: Good")
                elif rul > 30:
                    st.warning("⚠️ Condition: Monitor")
                else:
                    st.error("❌ Condition: Critical")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Model not loaded")