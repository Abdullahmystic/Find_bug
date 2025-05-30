from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = 'C:\\Users\\abdul\\Downloads\\updated_url_dataset.csv'  # Update path if needed
emails_dataset = pd.read_csv(file_path)

# Step 2: Handle missing values
emails_dataset['Data'] = emails_dataset['Data'].fillna('')
# Display the first few rows of the DataFrame

# Debugging Print - Total rows after handling missing values
print(f"Total rows after handling missing values: {len(emails_dataset)}")

# Step 3: Prepare features (X) and target (y)
X = emails_dataset['Data']
y = emails_dataset['Category']

# Step 4: Remove rare classes (before encoding)
class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index  # Keep only classes with >1 sample
emails_dataset = emails_dataset[emails_dataset['Category'].isin(valid_classes)]

# Update X and y after filtering
X = emails_dataset['Data']
y = emails_dataset['Category']

# Step 5: Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Debugging Print - Unique classes after encoding
print(f"Unique classes after encoding: {np.unique(y)}")

# Step 6: Convert text data to numeric using TF-IDF with improved parameters
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increased from 8000
    ngram_range=(1, 3),  # Extended from (1, 2) to capture more context
    sublinear_tf=True,
    stop_words='english',
    min_df=2,  # Remove extremely rare terms
    max_df=0.95  # Remove very common terms
)
X = vectorizer.fit_transform(X)

# Debugging Print - Shape of vectorized data
print(f"Shape of vectorized data: {X.shape}")

# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Debugging Prints - Training and Testing set sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 8: Train the Base Models with improved configurations
nb_model = MultinomialNB(alpha=0.1)  # Reduced alpha for less smoothing

xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    max_depth=7,  # Slightly increased
    learning_rate=0.05,  # Reduced for better generalization
    n_estimators=200,  # Increased from default
    subsample=0.8,  # Add subsampling to prevent overfitting
    colsample_bytree=0.8,  # Add column sampling
    random_state=42
)

# Add cross-validation to evaluate model stability
cv_scores_nb = cross_val_score(nb_model, X_train, y_train, cv=5)
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f"NB CV Score: {cv_scores_nb.mean():.4f} ± {cv_scores_nb.std():.4f}")
print(f"XGB CV Score: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

# Train the models
nb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Step 9: Get Predictions from Base Models
nb_preds = nb_model.predict_proba(X_train)
xgb_preds = xgb_model.predict_proba(X_train)

# Step 10: Stack Predictions as New Features
X_train_stacked = np.hstack((nb_preds, xgb_preds))
X_test_stacked = np.hstack((nb_model.predict_proba(X_test), xgb_model.predict_proba(X_test)))

# Normalize stacked features for better meta-classifier performance
scaler = StandardScaler()
X_train_stacked = scaler.fit_transform(X_train_stacked)
X_test_stacked = scaler.transform(X_test_stacked)

# Step 11: Train Meta-Classifier (Logistic Regression) with improved parameters
meta_classifier = LogisticRegression(
    C=1.0,  # Regularization strength
    max_iter=1000,  # Increased from 500
    solver='saga',  # Better for multi-class problems
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
meta_classifier.fit(X_train_stacked, y_train)

# Step 12: Predict using the Stacked Model
y_pred = meta_classifier.predict(X_test_stacked)

# Step 13: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Model Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report with updated labels
updated_class_names = label_encoder.inverse_transform(np.unique(y_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=updated_class_names, zero_division=0))

# Add ROC AUC score evaluation
try:
    # For multi-class problems
    y_pred_proba = meta_classifier.predict_proba(X_test_stacked)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc:.4f}")
except:
    # Skip if binary classification or too few samples per class
    pass

# Step 14: Confusion Matrix Visualization (Fixed Labeling Issue)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=updated_class_names,
            yticklabels=updated_class_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Add feature importance visualization for XGBoost
try:
    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) > 0:  # Only if features available
        plt.figure(figsize=(12, 8))
        top_n = 20  # Show top 20 features
        top_indices = np.argsort(xgb_model.feature_importances_)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = xgb_model.feature_importances_[top_indices]

        plt.barh(range(top_n), top_importances, align='center')
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Could not plot feature importance: {e}")


# Optional: Add learning curve function to diagnose overfitting/underfitting
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Uncomment the following line to generate learning curve (can be time-consuming)
# plot_learning_curve(meta_classifier, X_train_stacked, y_train, "Learning Curve (Meta Classifier)")
# plt.show()