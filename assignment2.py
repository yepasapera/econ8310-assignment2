import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Load the training data
data = pd.read_csv("assignment2train.csv")

# Preprocess the data: drop irrelevant columns and encode categorical ones
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Hour'] = data['DateTime'].dt.hour  # Extract hour from DateTime
data['DayOfWeek'] = data['DateTime'].dt.dayofweek  # Extract day of the week
data.drop(['id', 'DateTime'], axis=1, inplace=True)  # Drop 'id' and 'DateTime' columns

# Define target and features
Y = data['meal']
X = data.drop('meal', axis=1)

# Split the data into train and test sets
x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

# Standardize the data (scaling numeric values)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
xt_scaled = scaler.transform(xt)

# Initialize XGBoost classifier
xgb = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.5,
    objective='binary:logistic',  # binary classification
    eval_metric="logloss"
)

# Train the model
xgb.fit(x_scaled, y)

# Store the trained model
joblib.dump(xgb, 'modelFit.pkl')

# Make predictions
pred = xgb.predict(xt_scaled)

# Print the accuracy score
accuracy = accuracy_score(yt, pred)
print(f"Accuracy score: {accuracy*100:.2f}%")

# Optionally, print the classification report for more detailed metrics
print(classification_report(yt, pred))

# Load the test data for future predictions (use the same scaling steps)
test_data = pd.read_csv("assignment2test.csv")

# Preprocess the test data the same way as training data
test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['Hour'] = test_data['DateTime'].dt.hour
test_data['DayOfWeek'] = test_data['DateTime'].dt.dayofweek
test_data.drop(['id', 'DateTime'], axis=1, inplace=True)

# Ensure the test data columns match the training data columns (no 'meal' column)
X_test = test_data  # Here, we are only interested in the features

# Make sure the columns in X_test match the columns in X (training data)
X_test = X_test[X.columns]

# Scale the test data using the scaler from training
test_scaled = scaler.transform(X_test)

# Load the trained model (if needed)
modelFit = joblib.load('modelFit.pkl')

# Make predictions on the test data
test_pred = modelFit.predict(test_scaled)

# Print the columns of train and test data
print("Train columns:", X.columns)  # training features
print("Test columns:", X_test.columns)  # test features

# Print the predictions
print(test_pred)