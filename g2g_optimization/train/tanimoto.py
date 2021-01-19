import pandas as pd
import imp
import sys
import os
from rdkit.Chem import RDConfig
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

import sknetwork
from sknetwork.hierarchy import Ward
from sknetwork.hierarchy import cut_straight


def tan_adjacency(data):
    print('Calculating adjacency matrix...')
    start = time.time()
    size=len(data)
    adj = np.zeros((size,size))
    data['mol'] = data['SMILES'].apply(Chem.MolFromSmiles)
    data['fp'] = data['mol'].apply(Chem.RDKFingerprint)

    fp_pairs = [[i,j,(data['fp'].iloc[i],data['fp'].iloc[j])]  for i in range(size-1) for j in range(i,size)]
    fp_pairs = pd.DataFrame(fp_pairs)
    fp_pairs['dist'] = fp_pairs[2].apply(lambda x: DataStructs.FingerprintSimilarity(x[0],x[1]))
    
    adj = np.zeros((size,size))
    for i in range(len(fp_pairs)):
            x = fp_pairs[0].iloc[i]
            y = fp_pairs[1].iloc[i]
            adj[x,y],adj[y,x] = fp_pairs['dist'].iloc[i],fp_pairs['dist'].iloc[i]
            
    end = time.time()
    print('Time Elapsed: {}'.format(end-start))
    return adj
      

def adjacency_clusters(adj,n_clusters=None,threshold=None):
    print('Clustering molecules...')
    ward = Ward()
    dendrogram = ward.fit_transform(adj)
    labels = cut_straight(dendrogram,n_clusters=n_clusters,threshold=threshold)
    return labels