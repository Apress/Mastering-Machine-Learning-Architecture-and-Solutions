import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")
print(model["machine"])  # Vector representation of 'machine'
