# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  

# Load trained model
model = joblib.load("models/sentiment_pipeline_3class.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Make prediction
        prediction = model.predict([text])[0]

        # ✅ Convert NumPy int64 -> Python int -> string label
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map.get(int(prediction), "unknown")

        return jsonify({"sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
