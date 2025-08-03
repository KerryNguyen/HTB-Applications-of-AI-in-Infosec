import pandas as pd

# Set the file path to the dataset
file_path = r'skills_assessment_data/train.json'

# Read the json file into a DataFrame
df = pd.read_json(file_path)
print(df.head())

# Check for missing values
print("Missing values:\n", df.isnull().sum())
# Check for duplicates
print("Duplicate entries:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# Convert all text text to lowercase
df["text"] = df["text"].str.lower()
print("\n=== AFTER LOWERCASING ===")
print(df["text"].head(5))

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Remove non-essential punctuation and numbers, keep useful symbols like $ and !
df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["text"].head(5))

# Split each text into individual tokens
df["text"] = df["text"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["text"].head(5))

# Define a set of English stop words and remove them from the tokens
stop_words = set(stopwords.words("english"))
df["text"] = df["text"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOP WORDS ===")
print(df["text"].head(5))

# Stem each token to reduce words to their base form
stemmer = PorterStemmer()
df["text"] = df["text"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n=== AFTER STEMMING ===")
print(df["text"].head(5))

# Rejoin tokens into a single string for feature extraction
df["text"] = df["text"].apply(lambda x: " ".join(x))
print("\n=== AFTER JOINING TOKENS BACK INTO STRINGS ===")
print(df["text"].head(5))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
'''
# Initialize CountVectorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))

# Fit and transform the text column
X = vectorizer.fit_transform(df["text"])
'''
# Labels (target variable)
y = df["label"] # Converting labels to 1 (Positive) and 0 (Negative)

# Build the pipeline - create a fresh vectorizer inside the pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "classifier__C": [0.1, 1.0, 10.0, 100.0],  # 4 values instead of 12
    "classifier__class_weight": ['balanced'],  # 2 values
    "vectorizer__max_features": [20000],  # 2 values instead of 5
    "vectorizer__ngram_range": [(1, 3)],  # 2 values instead of 3
}

# More aggressive cross-validation
stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Perform the grid search with 5-fold cross-validation and the F1-score as metric
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=stratified_cv,
    scoring="accuracy", # Try accuracy instead of f1
    verbose=2
)

# Fit the grid search on the full dataset
grid_search.fit(df["text"], y)

# Extract the best model identified by the grid search
best_model = grid_search.best_estimator_
print("Best model parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Test
new_text_file_path = r'skills_assessment_data/test.json'
new_text_df = pd.read_json(new_text_file_path)

# Initialize preprocessing tools (same as training)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Preprocess function that mirrors the training-time preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s$!]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Check the structure of your test data
print("Test data columns:", new_text_df.columns.tolist())
print("Test data shape:", new_text_df.shape)
print("First few rows:")
print(new_text_df.head())

# Extract text column (adjust column name if needed)
if 'text' in new_text_df.columns:
    test_texts = new_text_df['text'].tolist()
else:
    # If test.json has a different structure, adjust accordingly
    print("Available columns:", new_text_df.columns.tolist())
    # You might need to change this based on your test data structure
    test_texts = new_text_df.iloc[:, 0].tolist()  # Use first column if no 'text' column

# Preprocess all texts
processed_texts = [preprocess_text(text) for text in test_texts]

# Transform preprocessed texts into feature vectors
X_new = best_model.named_steps["vectorizer"].transform(processed_texts)  # Fixed: was processed_text

# Predict with the trained classifier
predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)

# Display predictions and probabilities for each evaluated text
for i, original_text in enumerate(test_texts):
    prediction = "Positive" if predictions[i] == 1 else "Negative"
    positive_probability = prediction_probabilities[i][1]  # Probability of being positive
    negative_probability = prediction_probabilities[i][0]   # Probability of being negative
    
    # Show overall statistics instead of individual predictions
print(f"Total predictions: {len(predictions)}")
print(f"Positive predictions: {sum(predictions == 1)}")
print(f"Negative predictions: {sum(predictions == 0)}")
print(f"Percentage positive: {(sum(predictions == 1) / len(predictions)) * 100:.2f}%")

# Show confidence distribution
avg_confidence = np.mean(np.max(prediction_probabilities, axis=1))
print(f"Average prediction confidence: {avg_confidence:.3f}")

import joblib

# Save the trained model to a file for future use
model_filename = 'skills_assessment.joblib'
joblib.dump(best_model, model_filename)

print(f"Model saved to {model_filename}")
