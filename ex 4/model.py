import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# Load dataset
df = pd.read_csv('diabetes.csv')

# Define features and target
X = df.drop(columns=['Outcome'])  # Drop the target column to keep only features
y = df['Outcome']  # Define target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy before permutation: {accuracy * 100:.2f}%')

# Compute permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
feature_importances = perm_importance.importances_mean
feature_names = X.columns

# Print feature importance
for feature_name, importance in zip(feature_names, feature_importances):
    print(f'{feature_name}: {importance:.4f}')

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Permutation Importance")
plt.title("Permutation Feature Importance")
plt.show()
