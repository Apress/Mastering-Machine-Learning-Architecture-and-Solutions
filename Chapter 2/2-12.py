# Before Running the code for the first time you need to download NLTK tokenizer

import nltk
nltk.download('punkt_tab')

# Then proceed with tokenizer

from nltk.tokenize import word_tokenize

text = "Machine learning is fascinating."
tokens = word_tokenize(text)
print(tokens)
