import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('diabetes.csv')
df.dropna(axis=0, inplace=True)  # Drop missing values if any

# Define features and target
X = df.drop(columns=['Outcome'])  # Drop the target column to keep only features
y = df['Outcome']  # Define target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Training complete! Model accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, "diabetes_model.joblib")
