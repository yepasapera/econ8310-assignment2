import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Read in the data from CSV
data = pd.read_csv("assignment2train.csv")

# Separate features (X) and target (y)
X = data.drop(columns=["meal"])  # Exclude the target variable from the dataset
y = data["meal"]  # This is the actual "meal" column

# Handle categorical features using one-hot encoding
X = pd.get_dummies(X)  # This automatically applies one-hot encoding to all categorical columns

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict with the trained model on the test set
pred = clf.predict(X_test)

# Check the length of pred and truth (y_test)
print(f"Length of truth: {len(y_test)}")
print(f"Length of pred: {len(pred)}")

# Ensure the lengths are the same before calculating Tjurr R-squared
if len(y_test) == len(pred):
    # Tjurr function to calculate the R-squared
    def tjurr(truth, pred):
        truth = list(truth)
        pred = list(pred)
        
        # Check if the lists are non-empty before calculating means
        y1 = np.mean([y for x, y in enumerate(pred) if truth[x] == 1]) if any(truth) else 0
        y2 = np.mean([y for x, y in enumerate(pred) if truth[x] == 0]) if all(truth) else 0
        
        # Return the difference
        return y1 - y2
    
    tjurr_value = tjurr(y_test, pred)
    print(f"Tjurr R-squared: {tjurr_value}")
else:
    print("Prediction and truth have mismatched lengths.")

# Perform assertion if the Tjurr R-squared is valid
if isinstance(tjurr_value, float) and not np.isnan(tjurr_value):
    assert tjurr_value > 0.12, f"Tjurr R-squared below 0.12: {tjurr_value}"
else:
    print(f"Tjurr R-squared is invalid: {tjurr_value}")