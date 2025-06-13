import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model (Pipeline includes preprocessor and classifier)
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Define expected features in the correct order
feature_columns = [
    'job_title', 'department', 'seniority_level', 'tags_1', 'tags_2',
    'city', 'country', 'industry', 'company_city', 'company_country',
    'employee_count', 'annual_revenue_usd', 'founded_year', 'company_age', 'score', 'score_rating'
]

@app.route('/')
def home():
    return render_template('index.html')  # Optional: Web interface

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Create a DataFrame with the input data in the correct order
        input_data = pd.DataFrame([data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Return prediction as JSON
        return jsonify({'prediction': int(prediction), 'result': 'Yes' if prediction == 1 else 'No'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)