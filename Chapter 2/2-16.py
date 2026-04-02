from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
text_data = ["Machine learning is amazing", "Feature engineering is key to model performance"]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data
tfidf_features = vectorizer.fit_transform(text_data)
print(tfidf_features.toarray())

