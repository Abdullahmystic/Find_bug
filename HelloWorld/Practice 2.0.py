from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = 'C:\\Users\\abdul\\Downloads\\updated_url_dataset.csv'  # Update the path
emails_dataset = pd.read_csv(file_path)

# Step 2: Handle missing values
emails_dataset['Data'] = emails_dataset['Data'].fillna('')

# Debugging Print - Total rows after handling missing values
print(f"Total rows after handling missing values: {len(emails_dataset)}")

# Step 3: Prepare features (X) and target (y)
X = emails_dataset['Data']
y = emails_dataset['Category']

# Step 4: Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Ensure correct encoding

# Step 5: Remove rare classes (having only 1 sample)
unique, counts = np.unique(y, return_counts=True)
valid_classes = unique[counts > 1]  # Keep only classes with >1 occurrence
filtered_indices = np.isin(y, valid_classes)
X, y = X[filtered_indices], y[filtered_indices]

# Re-encode target labels to make them consecutive
y = LabelEncoder().fit_transform(y)

# Debugging Print - Unique classes after re-encoding
print(f"Unique classes after re-encoding: {np.unique(y)}")

# Step 6: Convert text data to numeric using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words='english')
X = vectorizer.fit_transform(X)

# Debugging Print - Shape of vectorized data
print(f"Shape of vectorized data: {X.shape}")

# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Debugging Prints - Training and Testing set sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 8: Train the Base Models
nb_model = MultinomialNB()
xgb_model = XGBClassifier( eval_metric='mlogloss')

nb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Step 9: Get Predictions from Base Models
nb_preds = nb_model.predict_proba(X_train)
xgb_preds = xgb_model.predict_proba(X_train)

# Step 10: Stack Predictions as New Features
X_train_stacked = np.hstack((nb_preds, xgb_preds))
X_test_stacked = np.hstack((nb_model.predict_proba(X_test), xgb_model.predict_proba(X_test)))

# Step 11: Train Meta-Classifier (Logistic Regression)
meta_classifier = LogisticRegression()
meta_classifier.fit(X_train_stacked, y_train)

# Step 12: Predict using the Stacked Model
y_pred = meta_classifier.predict(X_test_stacked)

# Step 13: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Model Accuracy: {accuracy * 100:.2f}%")

# Get the updated class names after filtering and re-encoding
updated_class_names = label_encoder.inverse_transform(np.unique(y_test))

# Use the updated class names in classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=updated_class_names, zero_division=0))

# Step 14: Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
