from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import vstack
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
y = label_encoder.fit_transform(y)

# Step 5: Convert text data to numeric using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features to reduce dimensionality
    sublinear_tf=True,
    stop_words='english'
)
X = vectorizer.fit_transform(X)
print(f"Shape of vectorized data: {X.shape}")

# Step 6: Handle classes with very few samples by duplicating efficiently
class_counts = pd.Series(y).value_counts()
rare_classes = class_counts[class_counts <= 1].index

# Efficient duplication of rare class samples
for rare_class in rare_classes:
    rare_indices = np.where(y == rare_class)[0]
    X_rare = X[rare_indices]
    y_rare = y[rare_indices]
    X = vstack([X, X_rare])  # Use sparse stacking
    y = np.concatenate([y, y_rare])

print(f"Total rows after handling rare classes: {len(y)}")

# Step 7: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 7: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 8: Compute class weights for imbalance handling
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Step 9: Define classifiers
lr = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced')
svc = LinearSVC(random_state=42, class_weight='balanced')
dt = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight=class_weights_dict)
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)

# Step 10: Define a hybrid VotingClassifier
hybrid_model = VotingClassifier(
    estimators=[('lr', lr), ('svc', svc), ('dt', dt), ('rf', rf)],
    voting='hard'
)

# Step 11: Train the hybrid model
hybrid_model.fit(X_train, y_train)

# Step 12: Predict and evaluate the model
y_pred = hybrid_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Hybrid Model Accuracy: {accuracy * 100:.2f}%")

# Step 13: Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 14: Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Step 15: Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 16: Save and inspect misclassified examples
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_examples = emails_dataset.iloc[misclassified_indices]
print("Misclassified Examples:")
print(misclassified_examples.head())
