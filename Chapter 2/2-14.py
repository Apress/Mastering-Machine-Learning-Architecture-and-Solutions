from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runner", "runs"]
stems = [stemmer.stem(word) for word in words]
print(stems)
