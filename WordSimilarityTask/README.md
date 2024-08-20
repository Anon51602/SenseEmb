
# Word Similarity Task Repository

This repository provides a pipeline for performing word similarity tasks on text data. 

## Input Requirements

- **Input File:** The input should be a `.txt` file where each line represents a separate paragraph.

## Pipeline Overview

To execute the word similarity task, follow these steps:

1. **Tokenize Target Words:**
   ```bash
   python a_tokenize_target_words.py
   ```


2. **Extract Embeddings with Aho-Corasick:**
   ```bash
   python b_getemb_AhoCorasick.py
   ```


3. **Cluster Tokens using K-Means:**
   ```bash
   python c_cluster_kmeans.py
   ```

4. **Evaluate the Clusters:**
   ```bash
   python d_max_evaluate.py
   ```


