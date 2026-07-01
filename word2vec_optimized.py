import multiprocessing
import os, requests, re
import nltk
import numpy as np
import gensim.models.word2vec as w2v
import sklearn.manifold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard.plugins as projector
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.plugins.projector import visualize_embeddings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("stopwords")

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return [word.lower() for word in words]

# Fetch text from Gutenberg
filepath = 'http://www.gutenberg.org/files/33224/33224-0.txt'
corpus_raw = requests.get(filepath).text

# Tokenize into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

# Clean and tokenize each sentence into words
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))
        
# Hyperparameters
num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1

# Initialize Word2Vec
# Define the exact path where you saved the model
model_path = os.path.join("trained", "sample", "model.w2v")

# Check if the file already exists
if os.path.exists(model_path):
    print(f"Saved model found at {model_path}. Loading...")
    
    # Load the model directly into memory
    model2vec = w2v.Word2Vec.load(model_path)
    
else:
    print("No saved model found. Initializing and training a new one...")
    
    # Initialize Word2Vec
    model2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling,
    )

    print("Building vocabulary...")
    model2vec.build_vocab(sentences)

    print("Starting training...")
    model2vec.train(sentences, total_examples=model2vec.corpus_count, epochs=100)

    # Save the model for next time
    if not os.path.exists(os.path.join("trained", 'sample')):
        os.makedirs(os.path.join("trained", 'sample'))

    model2vec.save(model_path)
    print(f"Model saved to {model_path}")

# Now you can proceed directly to using the model, skipping the wait!
print("Finding similar words to 'earth'...")
print(model2vec.wv.most_similar("earth"))

# ---------------------------------------------------------
# Plotting word clusters using the t-SNE algorithm
# ---------------------------------------------------------
print("Running t-SNE (this might take a minute)...")
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = model2vec.wv.vectors
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

# 2. Gensim 4.x fix: Use index_to_key instead of vocab
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in zip(model2vec.wv.index_to_key, all_word_vectors_matrix_2d)
    ],
    columns=["word", "x", "y"]
)

sns.set_context("poster")
ax = points.plot.scatter("x", "y", s=10, figsize=(20, 12))
fig = ax.get_figure()

# Show the plot so you can actually see the result!
plt.show()
print(points)

"""Visualize on Tensorboard"""
vocab_list = points.word.values.tolist()
embeddings = all_word_vectors_matrix

embedding_var = tf.Variable(all_word_vectors_matrix, dtype='float32', name='embedding')
projector_config = ProjectorConfig()

embedding = projector_config.embeddings.add()
embedding.tensor_name = embedding_var.name

log_dir = os.path.join("trained", "sample")

# 1. Save the metadata (the actual words) to a TSV file
metadata_path = os.path.join(log_dir, 'metadata.tsv')
with open(metadata_path, 'w', encoding='utf-8') as f:
    for word in model2vec.wv.index_to_key:
        f.write(f"{word}\n")

# 2. Save the weights (the 300D vectors) to a TSV file
tensors_path = os.path.join(log_dir, 'tensors.tsv')
np.savetxt(tensors_path, model2vec.wv.vectors, delimiter='\t')

# 3. Configure the TensorBoard Projector
config = ProjectorConfig()
embedding = config.embeddings.add()

# Tell the projector where to find the simple TSV files
embedding.tensor_path = 'tensors.tsv'
embedding.metadata_path = 'metadata.tsv'

# 4. Write the projector config to the log directory
visualize_embeddings(log_dir, config)

print(f"TensorBoard files saved successfully to {log_dir}!")