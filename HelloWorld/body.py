from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# file_path = 'C:\\Users\\abdul\\Downloads\\F_dataset.csv'  # Update path if needed
# emails_dataset = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
# print(emails_dataset.head())
import pandas as pd

# Step 1: Mount your Google Drive

# Step 3: Load your CSV file
# Replace this with the actual path to your CSV file in your Google Drive
file_path = 'C:\\Users\\abdul\\Downloads\\F_dataset.csv'

emails_dataset = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')

# Optional: Display the first few rows to see your column names
print(emails_dataset.head())

# Step 4: Check the number of classes and class names
# Replace 'label' with the actual name of your label column
label_column = 'label'  # Change this if your column is named differently

unique_classes = emails_dataset[label_column].unique()
num_classes = len(unique_classes)

print("Number of classes:", num_classes)
print("Class names:", unique_classes)
# # Step 2: Handle missing values
# emails_dataset['Data'] = emails_dataset['Data'].fillna('')
# # Display the first few rows of the DataFrame
#
# # Debugging Print - Total rows after handling missing values
# print(f"Total rows after handling missing values: {len(emails_dataset)}")
#
# # Step 3: Prepare features (X) and target (y)
# X = emails_dataset['Data']
# y = emails_dataset['Category']
#
# # Step 4: Remove rare classes (before encoding)
# class_counts = y.value_counts()
# valid_classes = class_counts[class_counts > 1].index  # Keep only classes with >1 sample
# emails_dataset = emails_dataset[emails_dataset['Category'].isin(valid_classes)]
#
# # Update X and y after filtering
# X = emails_dataset['Data']
# y = emails_dataset['Category']
#
# # Step 5: Encode target labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)
#
# # Debugging Print - Unique classes after encoding
# print(f"Unique classes after encoding: {np.unique(y)}")
#
# # Step 6: Convert text data to numeric using TF-IDF
# vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
# X = vectorizer.fit_transform(X)
#
# # Debugging Print - Shape of vectorized data
# print(f"Shape of vectorized data: {X.shape}")
#
# # Step 7: Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#
# # Debugging Prints - Training and Testing set sizes
# print(f"Training set size: {X_train.shape[0]}")
# print(f"Test set size: {X_test.shape[0]}")
#
# # Step 8: Train the Base Models
# nb_model = MultinomialNB(alpha=0.5)  # Adjusted smoothing parameter
# xgb_model = XGBClassifier(eval_metric='mlogloss', max_depth=6, learning_rate=0.1)
#
# nb_model.fit(X_train, y_train)
# xgb_model.fit(X_train, y_train)
#
# # Step 9: Get Predictions from Base Models
# nb_preds = nb_model.predict_proba(X_train)
# xgb_preds = xgb_model.predict_proba(X_train)
#
# # Step 10: Stack Predictions as New Features
# X_train_stacked = np.hstack((nb_preds, xgb_preds))
# X_test_stacked = np.hstack((nb_model.predict_proba(X_test), xgb_model.predict_proba(X_test)))
#
# # Normalize stacked features for better meta-classifier performance
# scaler = StandardScaler()
# X_train_stacked = scaler.fit_transform(X_train_stacked)
# X_test_stacked = scaler.transform(X_test_stacked)
#
# # Step 11: Train Meta-Classifier (Logistic Regression)
# meta_classifier = LogisticRegression(max_iter=500, solver='lbfgs')
# meta_classifier.fit(X_train_stacked, y_train)
#
# # Step 12: Predict using the Stacked Model
# y_pred = meta_classifier.predict(X_test_stacked)
#
# # Step 13: Evaluate Performance
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Stacked Model Accuracy: {accuracy * 100:.2f}%")
#
# # Generate a classification report with updated labels
# updated_class_names = label_encoder.inverse_transform(np.unique(y_test))
#
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=updated_class_names, zero_division=0))
#
# # Step 14: Confusion Matrix Visualization (Fixed Labeling Issue)
# conf_matrix = confusion_matrix(y_test, y_pred)
#
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=updated_class_names,
#             yticklabels=updated_class_names)
# plt.title('Confusion Matrix Heatmap')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
