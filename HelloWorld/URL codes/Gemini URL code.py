import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone # Needed for XGBoost in cross_val_predict sometimes

# --- Configuration ---
FILE_PATH = 'C:\\Users\\abdul\\Downloads\\updated_url_dataset.csv'  # <<< UPDATE THIS PATH
TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 5 # Number of folds for cross-validation prediction
MAX_FEATURES_TFIDF = 8000
NGRAM_RANGE_TFIDF = (1, 2)

# --- Step 1: Load the dataset ---
print(f"Loading dataset from: {FILE_PATH}")
try:
    emails_dataset = pd.read_csv(FILE_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please update the FILE_PATH variable.")
    exit()

# --- Step 2: Handle missing values ---
# Assuming 'Data' is the text column and 'Category' is the target
if 'Data' not in emails_dataset.columns or 'Category' not in emails_dataset.columns:
    print("Error: Dataset must contain 'Data' and 'Category' columns.")
    exit()

emails_dataset['Data'] = emails_dataset['Data'].fillna('')
print(f"Total rows before filtering rare classes: {len(emails_dataset)}")

# --- Step 3: Prepare initial features (X_text) and target (y_labels) ---
X_text = emails_dataset['Data']
y_labels = emails_dataset['Category']

# --- Step 4: Remove rare classes (before encoding and splitting) ---
print("Filtering rare classes...")
class_counts = y_labels.value_counts()
# Keep classes with at least CV_FOLDS samples for stratified CV, or minimum 2
min_samples_required = max(2, CV_FOLDS)
valid_classes = class_counts[class_counts >= min_samples_required].index
original_rows = len(emails_dataset)
emails_dataset = emails_dataset[emails_dataset['Category'].isin(valid_classes)]
rows_after_filtering = len(emails_dataset)
print(f"Removed {original_rows - rows_after_filtering} rows due to rare classes (< {min_samples_required} samples).")
print(f"Total rows after filtering rare classes: {rows_after_filtering}")

# Update X_text and y_labels after filtering
X_text = emails_dataset['Data']
y_labels = emails_dataset['Category']

if len(emails_dataset) == 0 or len(y_labels.unique()) < 2:
     print("Error: Not enough data or classes remaining after filtering rare classes.")
     exit()

# --- Step 5: Encode target labels ---
print("Encoding target labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
print(f"Target labels encoded into {len(label_encoder.classes_)} unique classes.")
# Debugging Print - Unique classes after encoding
print(f"Unique encoded classes: {np.unique(y)}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


# --- Step 6: Split dataset (Text data) ---
print(f"Splitting data into training ({1-TEST_SIZE:.0%}) and test ({TEST_SIZE:.0%}) sets...")
X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # Stratify based on encoded labels
)
print(f"Training set size: {len(X_text_train)}")
print(f"Test set size: {len(X_text_test)}")

# --- Step 7: Convert text data to numeric using TF-IDF (Fit ONLY on Train) ---
print("Vectorizing text data using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES_TFIDF,
    ngram_range=NGRAM_RANGE_TFIDF,
    sublinear_tf=True,
    stop_words='english'
)
# Fit on training data and transform training data
X_train = vectorizer.fit_transform(X_text_train)
# Transform test data using the *same* fitted vectorizer
X_test = vectorizer.transform(X_text_test)
print(f"Shape of vectorized training data (TF-IDF): {X_train.shape}")
print(f"Shape of vectorized test data (TF-IDF): {X_test.shape}")


# --- Step 8: Define Base Models ---
print("Defining base models (Naive Bayes, XGBoost)...")
nb_model = MultinomialNB(alpha=0.5)
# Ensure XGBoost uses settings compatible with predict_proba and potential cloning
xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False # Recommended to avoid deprecation warnings
)

# --- Step 9: Generate Out-of-Fold Predictions for Meta-Classifier Training ---
# Use StratifiedKFold for cross-validation predictions to maintain class distribution
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"Generating out-of-fold predictions for training set using {CV_FOLDS}-fold CV...")
# For Naive Bayes
nb_preds_oof = cross_val_predict(nb_model, X_train, y_train, cv=skf, method='predict_proba')

# For XGBoost (might need clone if predict_proba isn't directly supported or model state changes)
# Using clone ensures a fresh model for each fold in cross_val_predict
xgb_clone = clone(xgb_model) # Clone to be safe
xgb_preds_oof = cross_val_predict(xgb_clone, X_train, y_train, cv=skf, method='predict_proba')

print("Out-of-fold predictions generated.")
print(f"Shape of Naive Bayes OOF predictions: {nb_preds_oof.shape}")
print(f"Shape of XGBoost OOF predictions: {xgb_preds_oof.shape}")

# --- Step 10: Create Stacked Features ---
# Training stacked features from Out-of-Fold predictions
X_train_stacked = np.hstack((nb_preds_oof, xgb_preds_oof))
print(f"Shape of stacked training features: {X_train_stacked.shape}")

# Testing stacked features: Need predictions from base models trained on FULL training data
print("Retraining base models on full training data for test set predictions...")
nb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train) # Fit the original xgb_model instance

print("Generating predictions on the test set...")
nb_preds_test = nb_model.predict_proba(X_test)
xgb_preds_test = xgb_model.predict_proba(X_test)

X_test_stacked = np.hstack((nb_preds_test, xgb_preds_test))
print(f"Shape of stacked test features: {X_test_stacked.shape}")

# --- Step 11: Scale Stacked Features ---
print("Scaling stacked features...")
scaler = StandardScaler()
X_train_stacked_scaled = scaler.fit_transform(X_train_stacked)
X_test_stacked_scaled = scaler.transform(X_test_stacked)

# --- Step 12: Train Meta-Classifier (Logistic Regression) ---
print("Training meta-classifier (Logistic Regression)...")
meta_classifier = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE) # Increased max_iter, added random_state
meta_classifier.fit(X_train_stacked_scaled, y_train)
print("Meta-classifier trained.")

# --- Step 13: Predict using the Stacked Model ---
print("Predicting on the test set using the stacked model...")
y_pred = meta_classifier.predict(X_test_stacked_scaled)

# --- Step 14: Evaluate Performance ---
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Evaluation Results ---")
print(f"Stacked Model Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report with original labels
try:
    # Get unique labels present in y_test *and* y_pred to handle cases where some classes might be missing in either
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    class_names = label_encoder.inverse_transform(unique_labels)

    print("\nClassification Report:")
    # Use labels=unique_labels to ensure report matches the order of target_names
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=class_names, zero_division=0))
except ValueError as e:
     print(f"\nWarning: Could not generate classification report with full labels. Error: {e}")
     print("Classification Report (using encoded labels):")
     print(classification_report(y_test, y_pred, zero_division=0))
     class_names = [str(i) for i in np.unique(y_test)] # Fallback to encoded labels for heatmap

# --- Step 15: Confusion Matrix Visualization ---
print("\nGenerating Confusion Matrix Heatmap...")
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8)) # Adjusted size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,  # Use derived class names
            yticklabels=class_names)  # Use derived class names
plt.title('Confusion Matrix Heatmap - Stacked Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout() # Adjust layout
plt.show()

print("\n--- Script Finished ---")