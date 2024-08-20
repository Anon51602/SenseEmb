import os

import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import time
from tqdm import tqdm

import torch
import pickle

import argparse
from sklearn.cluster import MiniBatchKMeans



key_set = set()

import pickle

kmeans_dict = dict()
small_dict = dict()
paths = ["./wiki.pkl"]
for path in paths:
    print("Load",path)
    with open(path, 'rb') as handle:
        emb = pickle.load(handle)
    for word in emb:
        key_set.add(word)
        if isinstance(emb[word], list):
            data = np.stack(emb[word])
        elif isinstance(emb[word], torch.Tensor):
            data = np.array(emb[word])
        else:
            print(emb[word])
        if data.shape[0] < 5:
            if word in small_dict.keys():
                small_dict[word] = np.vstack([small_dict[word],data])
            else :
                small_dict[word]  = data
        else:
            if word in kmeans_dict.keys():
                kmeans_dict[word].partial_fit(data)
            else:
                kmeans_dict[word] = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=100)
                kmeans_dict[word].partial_fit(data)


output_dict = dict()
for key in key_set:
    if key in small_dict.keys():
        if key in kmeans_dict.keys():
            kmeans_dict[key].partial_fit(small_dict[key])
            output_dict[key] = kmeans_dict[key].cluster_centers_
        else:
            if small_dict[key].shape[0] < 5:
                output_dict[key] = small_dict[key]
            else:
                kmeans_dict[key] = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=100)
                kmeans_dict[key].partial_fit(small_dict[key])
                output_dict[key] = kmeans_dict[key].cluster_centers_
    else:
        output_dict[key] = kmeans_dict[key].cluster_centers_

# Save the cluster info
with open('./rare320K_large_5.pkl', 'wb') as handle:
    pickle.dump(output_dict, handle)

