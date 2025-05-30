from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = 'C:\\Users\\abdul\\Downloads\\combined_body_emails.csv'  # Update path if needed
emails_dataset = pd.read_csv(file_path)

# Step 2: Handle missing values
emails_dataset['text'] = emails_dataset['text'].fillna('')

# Step 3: Normalize and map label values to binary classes
safe_labels = ['Ham', 'ham', 'Safe Email']
phishing_labels = ['Phishing Email', 'Spam', 'spam']

emails_dataset['label'] = emails_dataset['label'].apply(
    lambda x: 0 if x in safe_labels else 1 if x in phishing_labels else None
)

# Drop rows with unknown labels
emails_dataset = emails_dataset.dropna(subset=['label'])

# Convert label to integer type
emails_dataset['label'] = emails_dataset['label'].astype(int)

# Debugging Print - Total rows after handling missing values and label mapping
print(f"Total rows after cleaning and mapping labels: {len(emails_dataset)}")

# Step 4: Prepare features (X) and target (y)
X = emails_dataset['text']
y = emails_dataset['label']

# Step 5: Convert text data to numeric using TF-IDF
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
X = vectorizer.fit_transform(X)

# Debugging Print - Shape of vectorized data
print(f"Shape of vectorized data: {X.shape}")

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Debugging Prints - Training and Testing set sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 7: Train the Base Models
nb_model = MultinomialNB(alpha=0.5)  # Adjusted smoothing parameter
xgb_model = XGBClassifier(eval_metric='logloss', max_depth=6, learning_rate=0.1)

nb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Step 8: Get Predictions from Base Models
nb_preds = nb_model.predict_proba(X_train)
xgb_preds = xgb_model.predict_proba(X_train)

# Step 9: Stack Predictions as New Features
X_train_stacked = np.hstack((nb_preds, xgb_preds))
X_test_stacked = np.hstack((nb_model.predict_proba(X_test), xgb_model.predict_proba(X_test)))

# Normalize stacked features
scaler = StandardScaler()
X_train_stacked = scaler.fit_transform(X_train_stacked)
X_test_stacked = scaler.transform(X_test_stacked)

# Step 10: Train Meta-Classifier (Logistic Regression)
meta_classifier = LogisticRegression(max_iter=500, solver='lbfgs')
meta_classifier.fit(X_train_stacked, y_train)

# Step 11: Predict using the Stacked Model
y_pred = meta_classifier.predict(X_test_stacked)

# Step 12: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
target_names = ['Safe Email (0)', 'Phishing Email (1)']
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Step 13: Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
