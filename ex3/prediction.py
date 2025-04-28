import sys
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Read input from command line and convert to numpy array
input_features = np.array(sys.argv[1:], dtype=float)

# Ensure the input has 8 features (Insert a placeholder 0 for missing "Glucose" if necessary)
if input_features.shape[0] == 7:
    input_features = np.insert(input_features, 0, 0)  # Insert 0 for missing "Glucose"

# Reshape the input to match the expected 2D shape
features_scaled = scaler.transform(input_features.reshape(1, -1))

# Make prediction
predicted_glucose = model.predict(features_scaled)

print(f"Predicted Glucose Level: {predicted_glucose[0]:.2f}")
