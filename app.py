from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
try:
    with open('elasticcv.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    print("Model and scaler loaded successfully!")
    print(f"Scaler expects {scaler.n_features_in_} features")
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
    model = None
    scaler = None

# Get the exact number of features expected by the scaler
EXPECTED_FEATURES = scaler.n_features_in_ if scaler else 43

# exact company names from training data
EXACT_COMPANY_NAMES = ['ASTON MARTIN', 'AUDI', 'Acura', 'BENTLEY', 'BMW', 'Bugatti', 'Cadillac', 'Chevrolet', 'FERRARI', 'Ford', 'GMC', 'HONDA', 'HYUNDAI', 'Jaguar Land Rover', 'Jeep', 'KIA', 'KIA  ', 'Kia', 'LAMBORGHINI', 'MAHINDRA', 'MARUTI SUZUKI', 'MERCEDES', 'Mazda', 'Mitsubishi', 'NISSAN', 'Nissan', 'Peugeot', 'Porsche', 'ROLLS ROYCE', 'ROLLS ROYCE ', 'TOYOTA', 'Tata Motors', 'Tesla', 'Toyota', 'VOLVO', 'Volkswagen', 'Volvo']

@app.route('/')
def index():
    return render_template('index.html', companies=sorted(EXACT_COMPANY_NAMES))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not loaded properly'}), 500
        
        # Get form data
        cc_capacity = float(request.form['cc_capacity'])
        horsepower = float(request.form['horsepower'])
        total_speed = float(request.form['total_speed'])
        performance = float(request.form['performance'])
        seats = int(request.form['seats'])
        torque = float(request.form['torque'])
        company = request.form['company']
        
        # Create feature array
        features = np.zeros(EXPECTED_FEATURES)
        features[0] = cc_capacity
        features[1] = horsepower  
        features[2] = total_speed
        features[3] = performance
        features[4] = seats
        features[5] = torque
        
        # Set company one-hot encoding
        try:
            company_index = EXACT_COMPANY_NAMES.index(company)
            features[6 + company_index] = 1.0
        except ValueError:
            return render_template('error.html', error=f"Company '{company}' not found in training data")
        
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return render_template('result.html', 
                             prediction=round(prediction, 2),
                             cc_capacity=cc_capacity,
                             horsepower=horsepower,
                             total_speed=total_speed,
                             performance=performance,
                             seats=seats,
                             torque=torque,
                             company=company)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

else:
    application = app  

