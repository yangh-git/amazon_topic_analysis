# BERTopic Pipeline

## Overview

The BERTopic pipeline is a modular, multi-step topic modeling framework that leverages modern NLP techniques to create interpretable topic representations from text data. The pipeline is highly customizable, allowing users to swap different models (e.g., UMAP for PCA, K-means for HDBSCAN) at each stage.

## The Six Steps of the BERTopic Pipeline

### 1. Extract Embeddings
Documents are converted into dense, high-dimensional vector representations (embeddings) that capture their semantic meaning using transformer-based models like BERT or Sentence-BERT. The default model for English is `all-MiniLM-L6-v2`.

### 2. Reduce Dimensionality
The high-dimensional embeddings are reduced to a lower dimension to facilitate effective clustering and mitigate the "curse of dimensionality". Uniform Manifold Approximation and Projection (UMAP) is the default method, but others like PCA can be used.

### 3. Cluster Reduced Embeddings
Documents with similar semantic meanings (close together in the reduced dimension space) are grouped into clusters. Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) is the default clustering algorithm, which is effective at identifying arbitrarily shaped clusters and flagging outliers as noise.

### 4. Tokenize Topics
The documents within each cluster are treated as a single, large "class" or topic. Words within each class are tokenized in preparation for representation modeling.

### 5. Extract Topic Words
The most important words for each cluster (topic) are identified using a technique called class-based TF-IDF (c-TF-IDF). This method highlights words that are frequent within a specific topic but rare across all topics.

### 6. (Optional) Fine-tune Topic Words
An additional, optional step uses representation models (like KeyBERT or even LLMs) to further refine or generate more descriptive topic labels beyond the c-TF-IDF keywords.

## Key Features

### Modular Design
Each component is independent and can be swapped out or customized to suit specific needs.

### Contextual Understanding
By using transformer embeddings, BERTopic captures the semantic context of words, which is a significant advantage over traditional methods like LDA or NMF.

### Interpretable Topics
The use of c-TF-IDF helps generate clear, human-interpretable topic descriptions.

### Scalability & Performance
The library is designed to work with large datasets and supports GPU acceleration for faster processing.

## Resources

For more details and documentation, you can visit the [official BERTopic GitHub Pages](https://maartengr.github.io/BERTopic/).
