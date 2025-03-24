import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Load the training data
train_data = pd.read_csv("assignment2train.csv")

# Check for any missing values
print(train_data.isnull().sum())

# Select features (products sold) and the target variable (meal purchase)
X = train_data.drop(columns=['id', 'DateTime', 'meal'])  # Excluding non-relevant columns
y = train_data['meal']  # Target variable

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the DecisionTreeClassifier with a max depth of 5 (you can adjust this)
model = DecisionTreeClassifier(max_depth=5)

# Train the model using the training data
model.fit(X_train, y_train)

# Save the model for future predictions
joblib.dump(model, 'meal_predictor_model.pkl')

# Predict the target on the test set
predictions = model.predict(X_test)

# Convert predictions to binary values (1 or 0)
pred = [1 if pred == 1 else 0 for pred in predictions]

# Evaluate the accuracy of the model
acc = accuracy_score(y_test, pred)
print("Model accuracy is {:.2f}%.".format(acc * 100))

# Load new test data (which should have the same features as the training data)
test_data = pd.read_csv("assignment2test.csv")

# Ensure that the test data has the same structure as the training data (drop 'id', 'DateTime')
X_new = test_data.drop(columns=['id', 'DateTime', 'meal'])  # Exclude 'meal' from test data

# Load the saved model and make predictions
model = joblib.load('meal_predictor_model.pkl')
new_predictions = model.predict(X_new)

# Output predictions (1 for meal, 0 for no meal)
print(new_predictions)