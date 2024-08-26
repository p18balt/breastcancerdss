import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('breast_cancer_predictor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Ensure all required fields are present
        required_fields = ['bmi', 'glucose', 'insulin', 'homa', 'leptin', 'adiponectin', 'resistin', 'mcp1', 'age']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing data fields'}), 400

        # Convert input data to a numpy array
        features = np.array([[
            float(data['bmi']),
            float(data['glucose']),
            float(data['insulin']),
            float(data['homa']),
            float(data['leptin']),
            float(data['adiponectin']),
            float(data['resistin']),
            float(data['mcp1']),
            float(data['age'])
        ]])

        # Make prediction
        prediction = model.predict(features)
        print(f"Prediction: {prediction[0]}")  # Debugging line

        # Map prediction to result
        result = 'Unhealthy' if prediction[0] == 1 else 'Healthy'
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
