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

# ============================================
# CUSTOM CSS - Brighter colors and vibrant text
# ============================================
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Main container */
    .main {
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers - brighter and bolder */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d6e 50%, #1a1a2e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #4a6fa5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    
    /* Section titles - brighter */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.8rem;
        margin-top: 0.2rem;
        border-left: 4px solid #4a6fa5;
        padding-left: 10px;
    }
    
    /* Input labels with units - brighter */
    .input-container {
        display: flex;
        flex-direction: column;
        margin-bottom: 0.2rem;
    }
    
    .input-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: #1a2a4a;
        margin-bottom: 2px;
    }
    
    .unit-badge {
        background: linear-gradient(135deg, #e8ecf1, #d5dce6);
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        color: #1a2a4a;
        font-weight: 700;
        border: 1px solid #c0c8d4;
    }
    
    /* Prediction result - brighter colors */
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d6e 50%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 0.8rem;
        box-shadow: 0 8px 32px rgba(26, 26, 46, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .rul-value {
        font-size: 3.2rem;
        font-weight: 900;
        margin: 0.2rem 0;
        background: linear-gradient(135deg, #ffffff, #a8b8d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(168, 184, 216, 0.3);
    }
    
    .rul-unit {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 600;
        color: #a8b8d8;
    }
    
    /* Status badges - brighter and more vibrant */
    .status-good {
        background: linear-gradient(135deg, #d4edda, #b7e4c7);
        color: #0d5e2e;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        font-size: 0.95rem;
        border: 2px solid #28a745;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    
    .status-monitor {
        background: linear-gradient(135deg, #fff3cd, #ffe69b);
        color: #6d4c00;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        font-size: 0.95rem;
        border: 2px solid #ffc107;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
    }
    
    .status-critical {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        font-size: 0.95rem;
        border: 2px solid #dc3545;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    
    /* Caption text - brighter */
    .stCaption {
        color: #4a6fa5 !important;
        font-weight: 500 !important;
    }
    
    /* Button styling - brighter */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d6e 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(26, 26, 46, 0.3) !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d2d6e 0%, #4a4a8e 100%) !important;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(26, 26, 46, 0.4) !important;
    }
    
    /* Success message - brighter */
    .stAlert {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    /* Info messages - brighter */
    .stAlert.info {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb) !important;
        color: #0c5460 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar - brighter text */
    .css-1d391kg, .css-1d391kg * {
        color: #1a2a4a !important;
    }
    
    /* Number input - brighter */
    .stNumberInput input {
        border-radius: 8px !important;
        border: 2px solid #c0c8d4 !important;
        padding: 0.4rem 0.6rem !important;
        font-weight: 600 !important;
        color: #1a2a4a !important;
        background: #f8f9fa !important;
    }
    
    .stNumberInput input:focus {
        border-color: #4a6fa5 !important;
        box-shadow: 0 0 0 3px rgba(74, 111, 165, 0.2) !important;
    }
    
    /* Slider labels - brighter */
    .stSlider label {
        font-weight: 700 !important;
        color: #1a2a4a !important;
    }
    
    /* Remove extra spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Metric styling - brighter */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fa, #e8ecf1);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #c0c8d4;
    }
    
    [data-testid="metric-container"] label {
        font-weight: 700 !important;
        color: #1a2a4a !important;
    }
    
    [data-testid="metric-container"] div {
        font-weight: 800 !important;
        color: #1a1a2e !important;
    }
    
    /* Footer - brighter */
    .css-15zrgzn {
        color: #4a6fa5 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling - brighter
st.markdown('<h1 class="main-header">Bearing RUL Prediction System</h1>', unsafe_allow_html=True)
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
        
        # Option to train model
        if st.button("🔄 Train Model Now", type="primary"):
            st.info("Training model... (this may take a moment)")
            
            try:
                # Generate sample data if no data exists
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
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error training model: {e}")

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = 'backend/models/bearing_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

model = load_model()

# ============================================
# PREDICTION INTERFACE - Brighter version
# ============================================

if model is not None:
    st.success("✅ Model ready for predictions")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sensor Input section
        st.markdown('<div class="section-title">📊 Sensor Input</div>', unsafe_allow_html=True)
        
        # Define features with their units
        feature_configs = [
            {'name': 'Vibration_X', 'label': 'Vibration X', 'unit': 'mm/s', 'default': 0.0, 'step': 0.1},
            {'name': 'Vibration_Y', 'label': 'Vibration Y', 'unit': 'mm/s', 'default': 0.0, 'step': 0.1},
            {'name': 'Vibration_Z', 'label': 'Vibration Z', 'unit': 'mm/s', 'default': 0.0, 'step': 0.1},
            {'name': 'Temperature', 'label': 'Temperature', 'unit': '°C', 'default': 25.0, 'step': 0.5},
            {'name': 'Pressure', 'label': 'Pressure', 'unit': 'bar', 'default': 1.0, 'step': 0.1},
            {'name': 'Speed', 'label': 'Speed', 'unit': 'RPM', 'default': 1500.0, 'step': 10.0},
        ]
        
        # Dictionary to store feature values
        features = {}
        
        # Create 3 columns for input fields
        cols = st.columns(3)
        for i, config in enumerate(feature_configs):
            with cols[i % 3]:
                # Display label with unit badge
                st.markdown(f"""
                <div class="input-container">
                    <div class="input-label">
                        <span>{config['label']}</span>
                        <span class="unit-badge">{config['unit']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                features[config['name']] = st.number_input(
                    label="",
                    value=config['default'],
                    format="%.2f",
                    key=config['name'],
                    step=config['step'],
                    help=f"Enter {config['label']} in {config['unit']}"
                )
        
        # Load simulation section
        st.markdown('<div class="section-title" style="margin-top: 1rem;">⚙️ Load Simulation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="input-container">
            <div class="input-label">
                <span>Load Factor</span>
                <span class="unit-badge">x multiplier</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        load_factor = st.slider(
            label="",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            format="%.1fx",
            help="Simulate different load conditions"
        )
        
        # Show current load status with brighter colors
        if load_factor < 0.5:
            st.info("🔹 **Light Load** - Bearing under minimal stress")
        elif load_factor < 0.8:
            st.info("🔸 **Normal Load** - Bearing operating as expected")
        elif load_factor < 1.5:
            st.warning("🔶 **Moderate Load** - Bearing experiencing increased stress")
        elif load_factor < 2.0:
            st.warning("🔺 **Heavy Load** - Bearing under significant stress")
        else:
            st.error("⛔ **Extreme Load** - Bearing stress critical!")
    
    with col2:
        # Prediction section
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        
        if st.button("Predict RUL", type="primary", use_container_width=True):
            try:
                # Prepare features
                feature_names = [c['name'] for c in feature_configs]
                base_features = np.array([features[name] for name in feature_names])
                
                # Apply load factor
                vibration_multiplier = 1.0 + (load_factor - 1.0) * 0.5
                temp_multiplier = 1.0 + (load_factor - 1.0) * 0.3
                pressure_multiplier = load_factor
                speed_multiplier = 1.0 + (load_factor - 1.0) * 0.1
                
                scaled_features = base_features.copy()
                scaled_features[0] *= vibration_multiplier
                scaled_features[1] *= vibration_multiplier
                scaled_features[2] *= vibration_multiplier
                scaled_features[3] *= temp_multiplier
                scaled_features[4] *= pressure_multiplier
                scaled_features[5] *= speed_multiplier
                
                scaled_features = scaled_features.reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(scaled_features)
                rul_hours = float(prediction[0])
                
                # Convert to days
                rul_days = max(0, min(rul_hours, 100))
                
                # Display RUL with brighter styling
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 600;">Remaining Useful Life</div>
                    <div class="rul-value">{rul_days:.1f}</div>
                    <div class="rul-unit">days</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=rul_days,
                    title={'text': "RUL", 'font': {'size': 14, 'color': '#1a2a4a', 'family': 'Inter, sans-serif'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'font': {'size': 32, 'color': '#1a1a2e', 'family': 'Inter, sans-serif'}, 'suffix': " days"},
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickwidth': 2,
                            'tickcolor': "#1a2a4a",
                            'tickfont': {'size': 11, 'color': '#1a2a4a', 'family': 'Inter, sans-serif'}
                        },
                        'bar': {
                            'color': "#2d2d6e",
                            'thickness': 0.3
                        },
                        'bgcolor': "#f0f2f6",
                        'borderwidth': 2,
                        'bordercolor': "#c0c8d4",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(220, 53, 69, 0.85)'},
                            {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.85)'},
                            {'range': [70, 100], 'color': 'rgba(40, 167, 69, 0.85)'}
                        ],
                        'threshold': {
                            'line': {'color': "#dc3545", 'width': 4},
                            'thickness': 0.6,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=15, r=15, t=25, b=15),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': "Inter, sans-serif"}
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Status message with brighter badges
                if rul_days > 70:
                    st.markdown('<span class="status-good">✅ Good Condition</span>', unsafe_allow_html=True)
                    st.caption("🟢 **Normal operation** - Continue monitoring")
                elif rul_days > 30:
                    st.markdown('<span class="status-monitor">⚠️ Monitor Condition</span>', unsafe_allow_html=True)
                    st.caption("🟡 **Schedule maintenance** - Bearing showing wear")
                else:
                    st.markdown('<span class="status-critical">❌ Critical Condition</span>', unsafe_allow_html=True)
                    st.caption("🔴 **Immediate maintenance** - Bearing failure imminent!")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    # Sidebar info - brighter
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ Model Info")
        st.info(f"""
        **Status:** ✅ Active
        **Type:** Random Forest
        **Features:** 6
        **RUL Unit:** Days
        """)
        
        # Load sample data button
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
    # Show instructions if no model
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
        - RUL in **days**
        - Health status (Good/Monitor/Critical)
        - Visual gauge indicator
        """)

# Footer - brighter
st.markdown("---")
st.caption("Built with ❤️ using Random Forest | Deployed on Streamlit Cloud")
