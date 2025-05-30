
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Specify the file paths for local execution
file_path = 'C:\\Users\\abdul\\Downloads\\malicious_phish.csv'  # Change this to your file's path

# Step 3: Load the CSV file using pandas
import pandas as pd
# Load the entire dataset
emails_dataset = pd.read_csv(file_path)

# Step 2: Handle missing values
emails_dataset['Data'] = emails_dataset['Data'].fillna('')
print(f"Total rows after handling missing values: {len(emails_dataset)}")

# Step 3: Prepare features (X) and target (y)
X = emails_dataset['Data']
y = emails_dataset['Category']

# Step 4: Convert text data to numeric using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=3000,  # Further reduce dimensionality
    sublinear_tf=True,
    stop_words='english'
)
X = vectorizer.fit_transform(X)
print(f"Shape of vectorized data: {X.shape}")

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]}")  # Number of rows in training set
print(f"Test set size: {X_test.shape[0]}")  # Number of rows in testing set


# Step 6: Define base models
lr = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
svc = LinearSVC(random_state=42)  # Faster alternative to SVC
dt = DecisionTreeClassifier(random_state=42, max_depth=10)

# Step 7: Create Voting Classifier
hybrid_model = VotingClassifier(
    estimators=[('lr', lr), ('svc', svc), ('dt', dt)],
    voting='hard'  # Hard voting for faster predictions
)

# Step 8: Train the model
hybrid_model.fit(X_train, y_train)
# Step 10: Predict and evaluate the model
y_pred = hybrid_model.predict(X_test)

# Step 11: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Hybrid Model Accuracy: {accuracy * 100:.2f}%")

# Step 12: Classification report for more detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 13: Confusion matrix to evaluate the model performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
