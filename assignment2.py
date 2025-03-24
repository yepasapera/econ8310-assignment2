import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("assignment2train.csv")

# Check column names to verify 'meal' exists
print("Columns in dataset:", data.columns)

# Check for missing values
print(f"Missing values in meal column: {data['meal'].isna().sum()}")

# Handle missing labels
data = data.dropna(subset=['meal'])  # Drop rows where meal is NaN

# Drop non-relevant columns
data = data.drop(columns=['id', 'DateTime'], errors='ignore')

# Define target and features
Y = data['meal']
X = data.drop('meal', axis=1)

# Convert categorical variables if any (not needed here, but good practice)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure all numeric

# Split data
x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

# Initialize model
model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.5, objective='binary:logistic')

# Train model
modelFit = model.fit(x, y)

# Predict on test data
pred = model.predict(xt)

# Load new test data for predictions
test_data = pd.read_csv("assignment2test.csv")
test_data = test_data.drop(columns=['id', 'DateTime'], errors='ignore')  # Drop unnecessary columns
test_data = test_data.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric

# Ensure test data columns match training features
missing_cols = set(X.columns) - set(test_data.columns)
extra_cols = set(test_data.columns) - set(X.columns)

print(f"Columns in training set but missing in test set: {missing_cols}")
print(f"Columns in test set but missing in training set: {extra_cols}")

# Add missing columns to test_data with default values (0)
for col in missing_cols:
    test_data[col] = 0

# Drop extra columns from test_data
test_data = test_data[X.columns]  # Reorder columns to match training set

# Make predictions for test set
test_predictions = model.predict(test_data)

# Make predictions for test set
test_predictions = model.predict(test_data)

# Print accuracy score
print(f"Accuracy score: {accuracy_score(yt, pred) * 100:.2f}%")

# Save predictions
pd.DataFrame({"meal_prediction": test_predictions}).to_csv("meal_predictions.csv", index=False)