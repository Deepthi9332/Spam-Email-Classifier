import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load data directly from a public URL link
url = "https://githubusercontent.com"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

# 2. Extract key features (Simulating sender/metadata features as requested)
# We track text length and the presence of digits/hyperlinks commonly seen in spam
df['length'] = df['text'].apply(len)
df['has_link'] = df['text'].str.contains('http|www').astype(int)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Process and Vectorize Email Text
vectorizer = CountVectorizer(stop_words='english')
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# 5. Train Machine Learning Classifier (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# 6. Predict and Evaluate Model Performance Metrics
predictions = model.predict(X_test_transformed)

# Map metric evaluations explicitly for printout
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions, pos_label='spam')
rec = recall_score(y_test, predictions, pos_label='spam')

print(f"--- Spam Email Classifier Evaluation Metrics ---")
print(f"Accuracy : {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}% (Tracks false positives - critical for spam!)")
print(f"Recall   : {rec * 100:.2f}% (Tracks percentage of caught spam)")
