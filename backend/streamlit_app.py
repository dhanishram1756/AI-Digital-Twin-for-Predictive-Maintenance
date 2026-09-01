import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bearing RUL Prediction",
    page_icon="🔧",
    layout="wide"
)

# ============================================
# CUSTOM CSS - Replicates your HTML design
# ============================================
st.markdown("""
<style>
    /* Import your fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main container */
    .main {
        padding: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Card-style containers - matches your HTML cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e8ecf1;
        margin-bottom: 1rem;
    }
    
    /* Input labels with units */
    .input-label {
        font-weight: 500;
        font-size: 0.9rem;
        color: #333;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .unit-badge {
        background: #f0f2f6;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Prediction result styling */
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
    }
    
    .rul-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .rul-unit {
        font-size: 1rem;
        opacity: 0.8;
    }
    
    /* Status badges */
    .status-good {
        background: #d4edda;
        color: #155724;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-monitor {
        background: #fff3cd;
        color: #856404;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-critical {
        background: #f8d7da;
        color: #721c24;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Gauge container */
    .gauge-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e8ecf1;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Button styling - matches your HTML buttons */
    .stButton > button {
        background: #1a1a2e !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: #2d2d44 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26,26,46,0.3);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e8ecf1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# REST OF YOUR APP CODE (with styled elements)
# ============================================

# Title with custom styling
st.markdown('<h1 class="main-header">🔧 Bearing RUL Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered predictive maintenance for bearing health monitoring</p>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Check if model exists
    model_path = 'backend/models/bearing_model.pkl'
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        st.success("✅ Model loaded")
    else:
        st.warning("⚠️ No model found")
        
        if st.button("🔄 Train Model Now", type="primary"):
            st.info("Training model... (this may take a moment)")
            
            try:
                data_path = 'backend/data/training_data.csv'
                
                if not os.path.exists(data_path):
                    st.info("Generating sample data...")
                    np.random.seed(42)
                    n_samples = 1000
                    X = np.random.randn(n_samples, 6)
                    y = 100 - (X[:, 0] * 20 + X[:, 1] * 15 + X[:, 2] * 10 + 
                              X[:, 3] * 5 + X[:, 4] * 5 + X[:, 5] * 3)
                    y = np.maximum(y, 0)
                    
                    os.makedirs('backend/data', exist_ok=True)
                    df = pd.DataFrame(X, columns=['Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                                                   'Temperature', 'Pressure', 'Speed'])
                    df['RUL'] = y
                    df.to_csv(data_path, index=False)
                    st.success("✅ Sample data created")
                
                df = pd.read_csv(data_path)
                X = df.drop('RUL', axis=1)
                y = df['RUL']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                os.makedirs('backend/models', exist_ok=True)
                joblib.dump(model, model_path)
                st.success("✅ Model trained and saved!")
                st.rerun()
                
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

# ============================================
# PREDICTION INTERFACE WITH UNITS
# ============================================

if model is not None:
    st.success("✅ Model ready for predictions")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Card-style container
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Sensor Input")
        
        # Input features with units
        features = {}
        feature_configs = [
            {'name': 'Vibration_X', 'label': 'Vibration X', 'unit': 'mm/s', 'value': 0.0},
            {'name': 'Vibration_Y', 'label': 'Vibration Y', 'unit': 'mm/s', 'value': 0.0},
            {'name': 'Vibration_Z', 'label': 'Vibration Z', 'unit': 'mm/s', 'value': 0.0},
            {'name': 'Temperature', 'label': 'Temperature', 'unit': '°C', 'value': 25.0},
            {'name': 'Pressure', 'label': 'Pressure', 'unit': 'bar', 'value': 1.0},
            {'name': 'Speed', 'label': 'Speed', 'unit': 'RPM', 'value': 1500.0},
        ]
        
        # Create 3 columns for input fields
        cols = st.columns(3)
        for i, config in enumerate(feature_configs):
            with cols[i % 3]:
                # Custom label with unit
                st.markdown(f"""
                <div class="input-label">
                    <span>{config['label']}</span>
                    <span class="unit-badge">{config['unit']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                features[config['name']] = st.number_input(
                    label="",  # Empty label since we're using custom HTML
                    value=config['value'],
                    format="%.2f",
                    key=config['name'],
                    step=0.1
                )
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
        
        # Load simulation slider
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚙️ Load Simulation")
        
        st.markdown("""
        <div class="input-label">
            <span>Load Factor</span>
            <span class="unit-badge">x multiplier</span>
        </div>
        """, unsafe_allow_html=True)
        
        load_factor = st.slider(
            label="",  # Empty label for custom styling
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            format="%.1fx"
        )
        
        if load_factor != 1.0:
            st.info(f"⚠️ Load factor {load_factor:.1f}x applied - sensor values will be scaled")
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Prediction")
        
        if st.button("🚀 Predict RUL", type="primary", use_container_width=True):
            try:
                # Prepare features
                feature_names = [c['name'] for c in feature_configs]
                feature_values = np.array([features[name] for name in feature_names])
                
                # Apply load factor
                feature_values = feature_values * load_factor
                feature_values = feature_values.reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(feature_values)
                rul = float(prediction[0])
                rul = max(0, min(rul, 100))
                
                # Display RUL in styled card
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size: 1rem; opacity: 0.8;">Remaining Useful Life</div>
                    <div class="rul-value">{rul:.1f}</div>
                    <div class="rul-unit">hours</div>
                </div>
                """, unsafe_allow_html=True)
                
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
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Status message
                if rul > 70:
                    st.markdown('<span class="status-good">✅ Good Condition</span>', unsafe_allow_html=True)
                    st.caption("🟢 Normal operation - Continue monitoring")
                elif rul > 30:
                    st.markdown('<span class="status-monitor">⚠️ Monitor Condition</span>', unsafe_allow_html=True)
                    st.caption("🟡 Increased vibration detected - Schedule maintenance soon")
                else:
                    st.markdown('<span class="status-critical">❌ Critical Condition</span>', unsafe_allow_html=True)
                    st.caption("🔴 Immediate maintenance required!")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ Model Info")
        st.info(f"""
        **Status:** ✅ Active
        **Type:** Random Forest
        **Features:** 6
        """)
        
        if st.button("📊 Load Sample Data"):
            sample = {
                'Vibration_X': 2.5,
                'Vibration_Y': 1.8,
                'Vibration_Z': 0.9,
                'Temperature': 75.0,
                'Pressure': 45.0,
                'Speed': 1500.0
            }
            for name, value in sample.items():
                if name in features:
                    features[name] = value
            st.success("✅ Sample values loaded!")
            st.rerun()

else:
    st.warning("""
    ### ⚠️ No trained model found
    
    **To get started:**
    1. Click **"Train Model Now"** in the sidebar
    2. Wait for training to complete (10-15 seconds)
    3. Start making predictions!
    """)
    
    with st.expander("📖 How it works"):
        st.markdown("""
        **This app predicts bearing Remaining Useful Life (RUL) using sensor data.**
        
        **Input features:**
        - Vibration X, Y, Z (mm/s)
        - Temperature (°C)
        - Pressure (bar)
        - Speed (RPM)
        
        **Output:**
        - RUL in hours
        - Health status (Good/Monitor/Critical)
        - Visual gauge indicator
        """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Random Forest | Deployed on Streamlit Cloud")
