# --------------------------------------------------
# Author Name : POTAKARI SMART
# USN         : 1AM23CI107
# Branch      : CSE - AIML
# College     : AMC Engineering College, Bengaluru
# Subject     : Deep Learning Laboratory
# --------------------------------------------------

"""
Program 1 (Word Embeddings)
Design and implement a neural network based model for generating
word embeddings for words in a document corpus.
"""

from gensim.models import Word2Vec

# Sample document corpus
corpus = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["deep", "learning"],
]

# Train the Word2Vec model
model = Word2Vec(
    sentences=corpus,
    vector_size=50,   # Size of word embeddings
    window=2,         # Context window size
    min_count=1,      # Include all words
    workers=1,
)

# Print the embedding vector for a word
print("Embedding for 'learning':")
print(model.wv["learning"])

# Find similar words
print("\nWords similar to 'learning':")
print(model.wv.most_similar("learning"))
