# Step 2: Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
file_path = 'C:\\Users\\abdul\\Downloads\\combined_body_emails.csv'  # Update path if needed
df = pd.read_csv(file_path)

# Step 4: Handle missing values in the 'text' column
df['text'] = df['text'].fillna('')
print(f"Total rows after handling missing values: {len(df)}")

# Step 5: Prepare features (X) and binary target (y)
X = df['text']
y_raw = df['label']

# Define label groupings
safe_keywords = ['Ham', 'ham', 'Safe Email']
phishing_keywords = ['Phishing Email', 'Spam', 'spam']

# Apply binary labeling: 0 for safe, 1 for phishing
y = y_raw.apply(lambda label: 0 if label in safe_keywords else 1)
num_classes = 2

# Step 6: Tokenize the text data
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index

# Step 7: Pad sequences to ensure equal length
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Step 8: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Convert labels to categorical for binary classification
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Step 9: Build the LSTM model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Step 10: Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 11: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=256
)

# Step 12: Predict and evaluate the model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"\nLSTM Model Accuracy: {accuracy * 100:.2f}%\n")

# Step 13: Classification Report
print("Classification Report:")
target_names = ['Safe Email (0)', 'Phishing Email (1)']
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# Step 14: Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()