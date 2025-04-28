from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
clf = joblib.load("diabetes_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    new_sample = np.array(data["data"]).reshape(1, -1)  # Ensure correct shape
    prediction = clf.predict(new_sample)
    return jsonify({"Diabetes Prediction": int(prediction[0])})

if __name__ == "__main__":
    print("Diabetes Prediction API is running!")
    app.run(host='0.0.0.0', port=5000)
