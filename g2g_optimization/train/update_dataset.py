import numpy as np
import pandas as pd
import argparse

from chemprop.data.scaffold import generate_scaffold

from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

from g2g_optimization.train.tanimoto import tan_adjacency,adjacency_clusters


def apply_screen(data,col_name,selection_type,selection_thresh,keep):
    data = data.sort_values(col_name,ascending=True)
    if selection_type=='Fraction':
        if keep=='High':
            data = data[-int(len(data)*selection_thresh):]
        elif keep=='Low':
            data = data[0:-int(len(data)*selection_thresh)]
        else:
            print('WARNING: INVALID KEEP TYPE')
    elif selection_type=='Cutoff':
        if keep=='High':
            data = data[data[col_name]>selection_thresh]
        elif keep=='Low':
            data = data[data[col_name]<selection_thresh]
        else:
            print('WARNING: INVALID KEEP TYPE')            
    else:
        print('WARNING: INVALID SELECTION TYPE')
        
    return data

def update_dataset(gen_evaluated,
                   data_file,
                   target='Target',
                   threshold=0.8, # Threshold improvement required to be kept (based on optimization target)
                   screen_file1=None,
                   selection_type1=None,
                   selection_thresh1=None,
                   keep1=None,
                   min_mol_wt=50, #(g/mol)
                   pairing_method='bemis_murcko',
                   n_clusters=None,
                   tan_threshold=None): 

    paired = pd.read_csv(gen_evaluated)

    # Remove molecules that do not follow screen 1:
    if screen_file1 is not None:
        screen1 = pd.read_csv(screen_file1)
        paired['Screen2_1']=paired['Mol2'].apply(lambda x: labeled[labeled[labeled.columns[0]]==x][labeled.columns[1]].iloc[0])
        paired = apply_screen(paired,'Screen2_1',selection_type1,selection_thresh1,keep1)

    # Remove Y molecules with low mw:
    paired['MolWt2'] = paired['Mol2'].apply(lambda x: ExactMolWt(Chem.MolFromSmiles(x)))
    paired = paired[paired['MolWt2']>min_mol_wt]

    # Remove molecules outside scaffold
    if pairing_method=='bemis_murcko':
        paired['Scaffold1'] = paired['Mol1'].apply(generate_scaffold)
        paired['Scaffold2'] = paired['Mol2'].apply(generate_scaffold)
        paired = paired[paired['Scaffold1']==paired['Scaffold2']]
    elif pairing_method=='tanimoto':
        mol_list = pd.concat((paired[['Mol1']].rename(columns={'Mol1':'SMILES'}),paired[['Mol2']].rename(columns={'Mol2':'SMILES'}))).drop_duplicates()
        adj = tan_adjacency(pd.DataFrame(mol_list))
        labels = adjacency_clusters(adj,n_clusters,threshold)
        
        mol_list['cluster'] = labels
        paired['Scaffold1'] = paired['Mol1'].apply(lambda x: mol_list[mol_list['SMILES']==x]['cluster'].values[0])
        paired['Scaffold2'] = paired['Mol2'].apply(lambda x: mol_list[mol_list['SMILES']==x]['cluster'].values[0])
        paired = paired[paired['Scaffold1']==paired['Scaffold2']]
    else:
        raise Exception('Unsupported pairing option:',pairing_method)

    # Make labeled dataset for input into next iteration:
    x_labeled = paired[['Mol1','Target1']].rename(columns={'Mol1':'SMILES','Target1':target})
    y_labeled = paired[['Mol2','Target2']].rename(columns={'Mol2':'SMILES','Target2':target})

    data_out = pd.concat([x_labeled,y_labeled]).drop_duplicates(subset=['SMILES'], keep='last')

    # Save data.csv file
    data_out.to_csv(data_file,index=False)

