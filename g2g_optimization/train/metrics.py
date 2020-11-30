import imp
import sys
import os
import numpy as np
import torch
import pandas as pd

from rdkit.Chem import RDConfig
from rdkit import Chem
from rdkit import DataStructs

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score')) 
import sascorer

# Statistics
def avg_improvement(df):
    avg = np.mean(df['Target2']-df['Target1'])
    return avg
def percent_improved(df):
    percent = np.mean([int(x>0) for x in df['Target2']-df['Target1']])
    return percent*100
def percent_improved_mae(df,mae=0.788):
    percent = np.mean([int(x>mae) for x in df['Target2']-df['Target1']])
    return percent*100
def avg_tanimoto(df):
    tanimoto = [DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(Chem.MolFromSmiles(x)),Chem.RDKFingerprint(Chem.MolFromSmiles(y))) for x,y in zip(df['Mol1'].values,df['Mol2'].values)]
    return np.mean(tanimoto)

def avg_improvement_no_duplicates(df):
    df = df[df['Mol1']!=df['Mol2']]
    avg = np.mean(df['Target2']-df['Target1'])
    return avg

def percent_improved_no_duplicates(df,mae=0.0):
    return np.mean([int(x) for x in (df['Mol1']!=df['Mol2']).values &(df['Target2']-df['Target1']>mae).values])*100
def percent_improved_mae_no_duplicates(df,mae=0.788):
    
    return np.mean([int(x) for x in (df['Mol1']!=df['Mol2']).values &(df['Target2']-df['Target1']>mae).values])*100

# Changes in SA Score:
def calc_avg_delta_sa(df):
    return np.mean(df['Mol2'].apply(lambda x: sascorer.calculateScore(Chem.MolFromSmiles(x)))-df['Mol1'].apply(lambda x: sascorer.calculateScore(Chem.MolFromSmiles(x))))

def percent_above_sa(df, threshold=3.5):
    return np.mean(df['Mol2'].apply(lambda x: sascorer.calculateScore(Chem.MolFromSmiles(x))>threshold))*100

# Percent failed to translate at all:
def translation_failure(df):
    return np.mean(df['Mol1']==df['Mol2'])*100

def sum_statistics(df):
    stats = {}
    stats['avg_improvment'] = avg_improvement(df)
    stats['percent_improved'] = percent_improved(df)
    stats['percent_improved_mae'] = percent_improved_mae(df)
    stats['avg_tanimoto'] = avg_tanimoto(df)
    stats['avg_improvement_no_duplicates'] = avg_improvement_no_duplicates(df)
    stats['percent_improved_no_duplicates'] = percent_improved_no_duplicates(df)
    stats['calc_avg_delta_sa'] = calc_avg_delta_sa(df)
    stats['percent_above_sa'] = percent_above_sa(df)
    stats['translation_failure'] = translation_failure(df)
    
    return stats