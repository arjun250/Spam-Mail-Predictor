from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "Spam Mail Prediction API is running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])  # Allow OPTIONS method
def predict():
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight response"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    try:
        data = request.get_json()
        email_text = data.get("email", "")

        if not email_text:
            return jsonify({"result": "Error: No email content provided"}), 400
        
        # Transform input using the saved TF-IDF vectorizer
        email_vectorized = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(email_vectorized)
        
        # Convert prediction to label (Spam or Ham)
        result = "Spam" if prediction[0] == 0 else "Not Spam"

        response = jsonify({"result": result})
        response.headers.add("Access-Control-Allow-Origin", "*")  # Allow frontend to access
        return response

    except Exception as e:
        response = jsonify({"result": f"Error: {str(e)}"})
        response.headers.add("Access-Control-Allow-Origin", "*")  # Ensure CORS for errors
        return response, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensure it runs on port 5000
