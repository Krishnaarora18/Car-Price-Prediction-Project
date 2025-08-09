import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarPricePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.load_models()
    
    def load_models(self):
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, 'elasticcv.pkl')
            scaler_path = os.path.join(base_path, 'scaler.pkl')

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError("Model or scaler file missing")

            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            
            with open(scaler_path, 'rb') as file:
                self.scaler = pickle.load(file)
            
            self.model_loaded = True
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.model_loaded = False
    
    def predict_price(self, features):
        if not self.model_loaded:
            raise ValueError("Models not loaded properly")
        
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            return max(0, round(prediction, 2))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Error making prediction: {str(e)}")

predictor = CarPricePredictionModel()

COMPANY_NAMES = [
    'ASTON MARTIN', 'AUDI', 'Acura', 'BENTLEY', 'BMW', 'Bugatti', 'Cadillac', 'Chevrolet',
    'FERRARI', 'Ford', 'GMC', 'HONDA', 'HYUNDAI', 'Jaguar Land Rover', 'Jeep', 'KIA', 'KIA  ',
    'Kia', 'LAMBORGHINI', 'MAHINDRA', 'MARUTI SUZUKI', 'MERCEDES', 'Mazda', 'Mitsubishi',
    'NISSAN', 'Nissan', 'Peugeot', 'Porsche', 'ROLLS ROYCE', 'ROLLS ROYCE ', 'TOYOTA',
    'Tata Motors', 'Tesla', 'Toyota', 'VOLVO', 'Volkswagen', 'Volvo'
]

@app.route('/')
def index():
    return render_template('index.html', 
                         brands=sorted(COMPANY_NAMES),
                         current_year=datetime.now().year,
                         model_status=predictor.model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        
        try:
            company = data.get('company', '').strip()
            cc_battery_capacity = float(data.get('cc_battery_capacity', 0))
            horsepower = float(data.get('horsepower', 0))
            total_speed = float(data.get('total_speed', 0))
            performance = float(data.get('performance', 0))
            seats = int(data.get('seats', 0))
            torque = float(data.get('torque', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid input data types'}), 400
        
        errors = []
        if not company or company not in COMPANY_NAMES:
            errors.append("Please select a valid car brand")
        if cc_battery_capacity <= 0 or cc_battery_capacity > 10000:
            errors.append("Please enter valid CC/Battery Capacity")
        if horsepower <= 0 or horsepower > 2000:
            errors.append("Please enter valid horsepower")
        if total_speed <= 0 or total_speed > 500:
            errors.append("Please enter valid top speed")
        if performance <= 0 or performance > 20:
            errors.append("Please enter valid 0-100 km/h time")
        if seats < 1 or seats > 10:
            errors.append("Please enter valid number of seats")
        if torque <= 0 or torque > 2000:
            errors.append("Please enter valid torque")
        
        if errors:
            return jsonify({'error': '; '.join(errors)}), 400
        
        if not predictor.model_loaded:
            return jsonify({'error': 'Prediction model not available'}), 503
        
        base_features = [
            cc_battery_capacity, horsepower, total_speed,
            performance, seats, torque
        ]
        
        company_features = [1.0 if brand == company else 0.0 for brand in sorted(COMPANY_NAMES)]
        features = base_features + company_features
        
        predicted_price = predictor.predict_price(features)
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': f"${predicted_price:,.2f}",
            'input_data': data,
            'prediction_id': f"PRED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

# Important for Gunicorn
application = app
