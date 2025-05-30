from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = 'C:\\Users\\abdul\\Downloads\\updated_url_dataset.csv'  # Update the file path
emails_dataset = pd.read_csv(file_path)

# Step 2: Handle missing values in the 'Data' column
emails_dataset['Data'] = emails_dataset['Data'].fillna('')
print(f"Total rows after handling missing values: {len(emails_dataset)}")

# Step 3: Prepare features (X) and target (y)
X = emails_dataset['Data']
y = emails_dataset['Category']  # Assuming 'Category' is the target column

# Step 4: Encode target labels if they are categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts category names into numeric values

# Step 5: Convert text data to numeric using Character-level TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features to reduce dimensionality
    sublinear_tf=True,
    analyzer='char_wb',  # Works well for URL-like data
    ngram_range=(1, 3)  # Capture character patterns (1-3 grams)
)
X = vectorizer.fit_transform(X)
print(f"Shape of vectorized data: {X.shape}")

# Step 6: Remove classes with only one sample before splitting
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts > 1].index  # Keep only classes with more than 1 sample

# Filter dataset to keep only valid classes
valid_indices = [i for i, label in enumerate(y) if label in valid_classes]
X = X[valid_indices]
y = y[valid_indices]

print(f"Total rows after filtering rare classes: {len(y)}")

# Step 7: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 8: Train a Multinomial Naïve Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Step 9: Predict and evaluate the model
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"MultinomialNB Accuracy: {accuracy * 100:.2f}%")

# Step 10: Classification report
print("Classification Report:")
# Ensure target_names matches the actual labels in y_test
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_[:len(np.unique(y_test))], zero_division=0))

# Step 11: Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Step 12: Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 13: Save and inspect misclassified examples
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_examples = emails_dataset.iloc[misclassified_indices]
print("Misclassified Examples:")
print(misclassified_examples.head())
