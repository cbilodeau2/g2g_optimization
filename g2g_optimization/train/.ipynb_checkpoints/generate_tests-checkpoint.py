import numpy as np
import sys
import os
import argparse
import pandas as pd
import time
from rdkit import Chem

from g2g_optimization.hgraph import *

def get_index_smiles(index_set,data_set):
    return [data_set[i].smiles for i in index_set]
            
def drop_dots(df):
    # Removes any molecular entries contianing more than one molecule
    
    return df[df['SMILES'].apply(lambda smiles: (bool(not '.' in smiles)))]

def drop_unusual_atoms(df):
    # Removes any molecular entries containing atoms not present in the common atom vocab
    
    def check_atoms(smiles):
        for atom in Chem.MolFromSmiles(smiles).GetAtoms():
            label = (atom.GetSymbol(), atom.GetFormalCharge())
            valid = True
            if not (label in common_atom_vocab.vocab):
                valid = False
                break
        return valid
    return df[df['SMILES'].apply(check_atoms)]
    
def remove_tails(df,target):
    df_sorted = df.sort_values(target)
    df_notails = df_sorted.iloc[int(len(df_sorted)/8):int(len(df_sorted)*7/8)]
    
    # Return without tails and shuffled:
    return df_notails.sample(frac=1)

def drop_invalid_smiles(df):
    for smile in df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smile)
        except:
            df = df[df['SMILES']!=smile]
    return df

def generate_test_sets(data_file,val_path,target='Solubility',cutoff= 0.8, sample_n=20,remove_tails_flag=True):  
    data = pd.read_csv(data_file)

    # Apply data filters:
    data = drop_dots(data)
    data = drop_unusual_atoms(data)
    data= drop_invalid_smiles(data)
    data= data[['SMILES',target]]
    
    if remove_tails_flag:
        data = remove_tails(data,target)
    
    data = data.sort_values(target)
    data['SMILES'] = data['SMILES'].apply(lambda x: x.replace('"', ''))
    
    
    data[0:25][['SMILES']].to_csv(os.path.join(val_path,'bottom.txt'),index=False,header=None)
    data[-25:][['SMILES']].to_csv(os.path.join(val_path,'top.txt'),index=False,header=None)
    print(data.iloc[int((len(data)*1)/4-12):int((len(data)*1)/4+13)][['SMILES']])
    data.iloc[int((len(data)*1)/4-12):int((len(data)*1)/4+13)][['SMILES']].to_csv(os.path.join(val_path,'med_bottom.txt'),index=False,header=None)
    data.iloc[int((len(data)*2)/4-12):int((len(data)*2)/4+13)][['SMILES']].to_csv(os.path.join(val_path,'medium.txt'),index=False,header=None)
    data.iloc[int((len(data)*3)/4-12):int((len(data)*3)/4+13)][['SMILES']].to_csv(os.path.join(val_path,'med_top.txt'),index=False,header=None)
    