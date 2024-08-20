import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import pickle
from tqdm import tqdm
import numpy as np
import os

from sklearn.cluster import KMeans
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output_keyword', type=str, default='./mrpc_minikmeans_5', help="Input and output keyword")
    parser.add_argument('--k', type=int, default=10, help="Number of Cluster")
    args = parser.parse_args()

    file_path = args.output_keyword  + ".pkl"
    if os.path.exists(file_path):
        print("Cache Load from ckpt!")
        with open(file_path, 'rb') as handle:
            save_data = pickle.load(handle)
        emb_cache_dict = save_data['emb_cache_dict']
        KMeans_dict = save_data['KMeans_dict']
    else:
        print("No ckpt!")
    
    output_dict = dict()
    for key in range(128000):
        if KMeans_dict[key] == None:
            # Not enough data
            if len(emb_cache_dict[key]) > 0:
                emb = np.vstack(emb_cache_dict[key])
                if emb.shape[0] <=args.k: 
                    output_dict[key] = torch.tensor(emb).half()
                else:
                    kmeans = KMeans(n_clusters=args.k, random_state=0,n_init="auto").fit(emb)
                    output_dict[key] = torch.tensor(kmeans.cluster_centers_).half()
            else:
                output_dict[key] = None
        else:
            output_dict[key] = torch.tensor(KMeans_dict[key].cluster_centers_).half()

    
    cluster_path = args.output_keyword  + "_cluster.pkl"
    with open(cluster_path, 'wb') as handle:
        pickle.dump(output_dict, handle)

        
    
