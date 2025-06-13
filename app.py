import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('logistic_regression_model.joblib')

# Define expected features in correct order
feature_columns = [
    'job_title', 'department', 'seniority_level', 'tags_1', 'tags_2',
    'city', 'country', 'industry', 'company_city', 'company_country',
    'employee_count', 'annual_revenue_usd', 'founded_year',
    'company_age', 'score', 'score_rating'
]

@app.route('/')
def home():
    return jsonify({'message': 'Prioritization Tool API is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Parse and convert input data
        input_data = {
            'job_title': data.get('job_title', ''),
            'department': data.get('department', ''),
            'seniority_level': data.get('seniority_level', ''),
            'tags_1': data.get('tags_1', ''),
            'tags_2': data.get('tags_2', ''),
            'city': data.get('city', ''),
            'country': data.get('country', ''),
            'industry': data.get('industry', ''),
            'company_city': data.get('company_city', ''),
            'company_country': data.get('company_country', ''),
            'employee_count': float(data.get('employee_count', 0)),
            'annual_revenue_usd': float(data.get('annual_revenue_usd', 0)),
            'founded_year': int(data.get('founded_year', 0)),
            'company_age': int(data.get('company_age', 0)),
            'score': float(data.get('score', 0)),
            'score_rating': data.get('score_rating', '')
        }

        # Create DataFrame in expected format
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability * 100, 2),
            'result': 'Yes' if prediction == 1 else 'No'
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
