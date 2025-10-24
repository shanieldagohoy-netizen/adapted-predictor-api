# app.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import json

app = Flask(__name__)

# Load the model, encoders, and column list when the app starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('ordinal_encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)
with open('target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)
with open('final_model_columns.json', 'r') as f:
    final_model_columns = json.load(f)

# Define the features for encoding
ordinal_features = ['Age', 'Education Level', 'Financial Condition', 'Load-shedding', 'Class Duration']
nominal_features = ['Gender', 'Location', 'Institution Type', 'IT Student', 'Internet Type', 'Network Type', 'Self Lms', 'Device']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert user input into a pandas DataFrame
        input_df_raw = pd.DataFrame([data])
        
        # Preprocess the user's input EXACTLY like the training data
        input_df_processed = input_df_raw.copy()
        input_df_processed[ordinal_features] = ordinal_encoder.transform(input_df_processed[ordinal_features])
        input_df_processed = pd.get_dummies(input_df_processed, columns=nominal_features, drop_first=True)
        
        # Align columns to match the model's training data
        input_final = input_df_processed.reindex(columns=final_model_columns, fill_value=0)
        
        # Make the prediction
        prediction_numeric = model.predict(input_final)
        
        # Decode the prediction back to a human-readable label
        prediction_label = target_encoder.inverse_transform(prediction_numeric.reshape(-1, 1))
        
        # Return the result as JSON
        return jsonify({'prediction': prediction_label[0][0]})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # For local testing, not used by Render
    app.run(port=5000, debug=True)