\# 🔧 Bearing RUL Prediction System



AI-powered predictive maintenance system for bearing health monitoring using Random Forest regression.



\## 🚀 Features



\- \*\*Real-time RUL Prediction\*\*: Predict remaining useful life from sensor data

\- \*\*Interactive Dashboard\*\*: Modern, responsive web interface

\- \*\*Load Simulation\*\*: Simulate load increase effects on bearing life

\- \*\*Batch Prediction\*\*: Process multiple samples at once

\- \*\*REST API\*\*: Easy integration with other systems

\- \*\*History Tracking\*\*: Keep track of all predictions



\## 📊 Dashboard Preview



!\[Dashboard](https://via.placeholder.com/800x400?text=Bearing+RUL+Dashboard)



\## 🛠️ Tech Stack



\- \*\*Backend\*\*: Flask, Python

\- \*\*Frontend\*\*: HTML5, CSS3, JavaScript

\- \*\*ML Model\*\*: Random Forest (scikit-learn)

\- \*\*Deployment\*\*: Docker, Gunicorn



\## 📁 Project Structure

.

├── backend/ # Flask API server

│ ├── app.py # Main application

│ ├── train\_model.py # Model training script

│ ├── models/ # Saved models

│ ├── data/ # Training data

│ └── templates/ # HTML templates

├── frontend/ # Standalone frontend

├── docs/ # Documentation

└── tests/ # Test scripts





\## 🚀 Quick Start



\### Prerequisites

\- Python 3.8+

\- pip



\### Installation



```bash

\# Clone the repository

git clone https://github.com/your-username/bearing-rul-system.git

cd bearing-rul-system



\# Install dependencies

pip install -r backend/requirements.txt



\# Train the model

cd backend

python train\_model.py



\# Run the server

python app.py



\# Open browser at http://localhost:8000



