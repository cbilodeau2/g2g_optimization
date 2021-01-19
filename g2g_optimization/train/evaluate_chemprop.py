import imp
import sys
import os
import numpy as np
import torch
import pandas as pd

from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader,MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit
from chemprop.train.predict import predict

from rdkit.Chem import RDConfig
from rdkit import Chem
from rdkit import DataStructs

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score')) 
import sascorer

from g2g_optimization.train.metrics import *

def evaluate_chemprop(decoded_path,fold_path,chemprop_path):

    data = pd.read_csv(decoded_path,header=None,delimiter=' ')
#    data = data.rename(columns={0:'Mol1',1:'Mol2'})

#     device = torch.device('cuda')
#     model = load_checkpoint(fold_path,device=device)
#     scaler, features_scaler = load_scalers(fold_path)

#     smiles1 = list(data['Mol1'])
#     print('Loading data')
#     full_data = get_data_from_smiles(
#         smiles=smiles1,
#         skip_invalid_smiles=False,
#         features_generator=None
#     )

#     test_data = MoleculeDataset(full_data)
#     test_data_loader=MoleculeDataLoader(dataset=test_data)

#     model_preds1 = predict(
#                 model=model,
#                 data_loader=test_data_loader,
#                 scaler=scaler)

#     smiles2 = list(data['Mol2'])
#     print('Loading data')
#     full_data = get_data_from_smiles(
#         smiles=smiles2,
#         skip_invalid_smiles=False,
#         features_generator=None
#     )

#     test_data = MoleculeDataset(full_data)
#     test_data_loader=MoleculeDataLoader(dataset=test_data)

#     model_preds2 = predict(
#                 model=model,
#                 data_loader=test_data_loader,
#                 scaler=scaler)
    
    temp_folder='tmp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
    
    data[0].to_csv(os.path.join(temp_folder,'col1.csv'),index=False)
    data[1].to_csv(os.path.join(temp_folder,'col2.csv'),index=False)
    
    os.system('python '+os.path.join(chemprop_path,'predict.py')+' --test_path '+os.path.join(temp_folder,'col1.csv')+' --batch_size 16 --checkpoint_dir '+fold_path+' --preds_path '+os.path.join(temp_folder,'preds_col1.csv'))
    
    os.system('python '+os.path.join(chemprop_path,'predict.py')+' --test_path '+os.path.join(temp_folder,'col2.csv')+' --batch_size 16 --checkpoint_dir '+fold_path+' --preds_path '+os.path.join(temp_folder,'preds_col2.csv'))

    preds1 = pd.read_csv(os.path.join(temp_folder,'preds_col1.csv'))
    preds1 = preds1.rename(columns={"0":"Mol1",preds1.columns[1]:"Target1"})
    preds2 = pd.read_csv(os.path.join(temp_folder,'preds_col2.csv'))
    preds2 = preds2.rename(columns={"1":"Mol2",preds2.columns[1]:"Target2"})
    preds_tot = pd.concat((preds1,preds2),axis=1)

#     preds_tot = pd.DataFrame()
#     preds_tot['Mol1'] = smiles1
#     preds_tot['Target1'] = [x[0] for x in model_preds1]
#     preds_tot['Mol2'] = smiles2
#     preds_tot['Target2'] = [x[0] for x in model_preds2]

    statistics = sum_statistics(preds_tot)
    return statistics,preds_tot

def evaluate_chemprop_onecol(data,fold_path,chemprop_path):
    temp_folder='tmp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        
    data.to_csv(os.path.join(temp_folder,'temp.csv'),index=False)
    os.system('python '+os.path.join(chemprop_path,'predict.py')+' --test_path '+os.path.join(temp_folder,'temp.csv')+' --checkpoint_dir '+fold_path+' --preds_path '+os.path.join(temp_folder,'preds_temp.csv') + ' > /dev/null')
    preds = pd.read_csv(os.path.join(temp_folder,'preds_temp.csv'))
    
    return preds

def evaluate_chemprop_sol(decoded_path,solvent,fold_path,chemprop_path):

    data = pd.read_csv(decoded_path,header=None,delimiter=' ')
    temp_folder='tmp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        
        
    data['sol'] = solvent
    data[[0,'sol']].to_csv(os.path.join(temp_folder,'col1.csv'),index=False)
    data[[1,'sol']].to_csv(os.path.join(temp_folder,'col2.csv'),index=False)
    
    os.system('python '+os.path.join(chemprop_path,'predict.py')+' --test_path '+os.path.join(temp_folder,'col1.csv')+' --checkpoint_dir '+fold_path+' --preds_path '+os.path.join(temp_folder,'preds_col1.csv')+' --number_of_molecules 2')
    
    os.system('python '+os.path.join(chemprop_path,'predict.py')+' --test_path '+os.path.join(temp_folder,'col2.csv')+' --checkpoint_dir '+fold_path+' --preds_path '+os.path.join(temp_folder,'preds_col2.csv')+' --number_of_molecules 2')

    preds1 = pd.read_csv(os.path.join(temp_folder,'preds_col1.csv'))
    preds1 = preds1.rename(columns={"0":"Mol1",preds1.columns[2]:"Target1"})
    preds2 = pd.read_csv(os.path.join(temp_folder,'preds_col2.csv'))
    preds2 = preds2.rename(columns={"1":"Mol2",preds2.columns[2]:"Target2"})
    preds_tot = pd.concat((preds1,preds2),axis=1)

    statistics = sum_statistics(preds_tot)
    return statistics,preds_tot    
    