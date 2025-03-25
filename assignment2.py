import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# training data
data = pd.read_csv("assignment2train.csv")

# process and encode data, extract datetime
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Hour'] = data['DateTime'].dt.hour 
data['DayOfWeek'] = data['DateTime'].dt.dayofweek  
data.drop(['id', 'DateTime'], axis=1, inplace=True)

# target
Y = data['meal']
X = data.drop('meal', axis=1)

# Split data as week 6
x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
xt_scaled = scaler.transform(xt)

# XGBoost classifier
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric="logloss",
    scale_pos_weight=1
)

# Store the model
model = xgb
# Train model
model.fit(x_scaled, y)

# Store trained model
joblib.dump(model, 'modelFit.pkl')

#predict
pred = model.predict(xt_scaled)

# Print the accuracy score
accuracy = accuracy_score(yt, pred)
print(f"Accuracy score: {accuracy*100:.2f}%")

#print the classification report
#print(classification_report(yt, pred))

# Load test data
test_data = pd.read_csv("assignment2test.csv")

# process test data
test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['Hour'] = test_data['DateTime'].dt.hour
test_data['DayOfWeek'] = test_data['DateTime'].dt.dayofweek
test_data.drop(['id', 'DateTime'], axis=1, inplace=True)

# column test to test data
X_test = test_data

#x_test equals test datak, scale test data
X_test = X_test[X.columns]
test_scaled = scaler.transform(X_test)

# Load the trained model
modelFit = joblib.load('modelFit.pkl')

# predict
test_pred = modelFit.predict(test_scaled)

pred = model.predict(test_scaled)

# Make sure the predictions are a list or numpy array with numeric values
if isinstance(pred, np.ndarray):  # If predictions are numpy array
    pred = pred.tolist()  # Convert to list if necessary

# Print the columns of train and test data
#print("Train columns:", X.columns) 
#print("Test columns:", X_test.columns)

# Print the predictions
print(test_pred)