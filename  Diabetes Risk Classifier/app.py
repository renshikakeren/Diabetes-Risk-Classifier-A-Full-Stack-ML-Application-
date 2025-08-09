import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please run src/model_training.py first.")
    model = None
    scaler = None


@app.route('/')
def index():
    """Renders the main page with the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the web form.
    It takes user input, preprocesses it, and returns a prediction.
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        # Get data from the form
        data = request.form.to_dict()
        data = {key: float(value) for key, value in data.items()}

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])

        # Add the engineered feature 'BMI_Category'
        def get_bmi_category(bmi):
            if bmi < 18.5:
                return 0
            elif 18.5 <= bmi < 24.9:
                return 1
            elif 24.9 <= bmi < 29.9:
                return 2
            else:
                return 3

        input_df['BMI_Category'] = input_df['BMI'].apply(get_bmi_category)

        # Scale the input data using the same scaler used during training
        # Ensure the order of columns matches the training data
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age', 'BMI_Category']
        input_scaled = scaler.transform(input_df[feature_names])

        # Make a prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1] * 100

        result = "Positive" if prediction == 1 else "Negative"
        message = f"The model predicts the patient is **{result}** for diabetes with a probability of **{prediction_proba:.2f}%**."

        return jsonify({'prediction': result, 'message': message})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
