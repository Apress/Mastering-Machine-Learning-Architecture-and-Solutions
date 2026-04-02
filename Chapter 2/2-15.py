# If you have not downloaded yet, download wordnet first
import nltk

nltk.download('wordnet')

# Then proceed with the following
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "runner", "runs"]
lemmas = [lemmatizer.lemmatize(word, pos="v") for word in words]
print(lemmas)

