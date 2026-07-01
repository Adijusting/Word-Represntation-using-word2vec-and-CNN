# NLP Text Processing and Classification Pipeline

This repository contains Python scripts for Natural Language Processing (NLP) tasks, specifically focusing on generating word embeddings from raw text and constructing a Convolutional Neural Network for text classification.

## Repository Contents

* `word2vec_optimized.py`: A complete pipeline for fetching text data, preprocessing, training word embeddings using Gensim's Word2Vec, and visualizing the high-dimensional vectors.
* `text_cnn.py`: A TensorFlow implementation of a Text Convolutional Neural Network (TextCNN) architecture designed for sentence classification.

---

## 1. Word Embeddings and Visualization (`word2vec_optimized.py`)

This script downloads a public domain book from Project Gutenberg, processes the text, and trains a Skip-gram Word2Vec model. It then provides tools to visualize the semantic relationships between words.

### Features:
* **Data Ingestion:** Automatically fetches raw text using the `requests` library.
* **Preprocessing:** Utilizes `nltk` for sentence and word tokenization (handling the `punkt` and `punkt_tab` resources).
* **Model Training:** Trains a 300-dimensional Word2Vec model using `gensim` (multiprocessed for efficiency).
* **State Management:** Saves and loads trained models (`trained/sample/model.w2v`) to avoid redundant training times.
* **2D Visualization:** Reduces the 300D embeddings to 2D using `sklearn.manifold.TSNE` and plots them using `seaborn` and `matplotlib`.
* **TensorBoard Integration:** Exports the embeddings and vocabulary metadata to `.tsv` formats compatible with the TensorBoard Embedding Projector.

### Usage:
Run the script directly. If it is the first run, it will train the model. Subsequent runs will load the saved model from disk.
```bash
python word2vec_optimized.py
