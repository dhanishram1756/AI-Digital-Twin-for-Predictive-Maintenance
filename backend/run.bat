@echo off
:: run.bat - Startup script for Windows

echo =====================================
echo 🚀 BEARING RUL PREDICTION SYSTEM
echo =====================================

if not exist "models\bearing_model_real.pkl" (
    echo ⚠️ Model not found. Training new model...
    python train_model.py
)

echo.
echo 🌐 Starting server at http://localhost:8000
echo    Press Ctrl+C to stop
echo =====================================
python app.py