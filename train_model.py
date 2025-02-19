import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
raw_mail_data = pd.read_csv('spam_updated.csv')

# Replace missing values with empty string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Encode labels (Spam = 0, Ham = 1)
mail_data['v1'] = mail_data['v1'].map({'spam': 0, 'ham': 1})

# Verify dataset columns
print(mail_data.head())

# Separate features and labels
X = mail_data['v2']  # Email text
Y = mail_data['v1']  # Labels

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform text data into numerical features using TF-IDF
feature_extraction = TfidfVectorizer(min_df=5, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert labels to integers
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluate Model
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print(f"Training Accuracy: {accuracy_on_training_data:.2f}")
print(f"Testing Accuracy: {accuracy_on_test_data:.2f}")

# Test with a sample email
input_mail = ["Lottery Scam: You have won $1,000,000! Click now!"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('✅ Not Spam (Ham)')
else:
    print('🚨 Spam Detected!')

# Save the trained model and vectorizer
joblib.dump(feature_extraction, "vectorizer.pkl")
joblib.dump(model, "spam_model.pkl")

print("✅ Model and vectorizer saved successfully!")
