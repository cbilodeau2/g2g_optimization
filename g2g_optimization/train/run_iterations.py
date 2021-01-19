import argparse
import rdkit
import os

from g2g_optimization.train.run_training import run_training
from g2g_optimization.train.decode import decode
from g2g_optimization.hgraph import common_atom_vocab
from g2g_optimization.train.args import read_args
from g2g_optimization.train.evaluate_chemprop import evaluate_chemprop,evaluate_chemprop_sol
from g2g_optimization.train.update_dataset import update_dataset

def iterate_round(args_file,save_dir,data_path,chemprop_path,constraint_file=None,iteration_num=1,solvent=None):

    args = read_args(args_file)
    save_dir1 = os.path.join(save_dir,'iteration'+str(iteration_num))
    if not os.path.isdir(save_dir1):
        os.mkdir(save_dir1) 
        
    # Train model:
    run_training(data_path,save_dir1,args_file,chemprop_path=chemprop_path,constraint_file=constraint_file)

    # Make augment folder
    if not os.path.isdir(os.path.join(save_dir1,'augment')):
        os.mkdir(os.path.join(save_dir1,'augment'))
    augment_folder = os.path.join(save_dir1,'augment')

    molfile = os.path.join(save_dir1,'inputs','mols.txt')
    vocab = os.path.join(save_dir1,'inputs','vocab.txt')
    model = os.path.join(save_dir1,'models','model.'+str(args['epoch']-1))
    args_file = os.path.join(save_dir1,'input.dat')

    # Generate new molecules based on original dataset:
    decode(molfile,
           vocab,
           model,
           os.path.join(augment_folder,'gen_out.csv'),
           args, # Args are needed to make sure the network architecture is correct
           atom_vocab=common_atom_vocab,
           num_decode = args['num_decode'])

    # Assign/predict molecule properties:
    if solvent == None:
        _,preds_tot = evaluate_chemprop(os.path.join(augment_folder,'gen_out.csv'),fold_path=args['fold_path'],chemprop_path=chemprop_path)
        preds_tot.to_csv(os.path.join(augment_folder,'gen_evaluated.csv'),index=False)
    else:
        _,preds_tot = evaluate_chemprop_sol(os.path.join(augment_folder,'gen_out.csv'),solvent=solvent,fold_path=args['fold_path'],chemprop_path=chemprop_path)
        preds_tot.to_csv(os.path.join(augment_folder,'gen_evaluated.csv'),index=False)
        
    # Apply filters and create new datafile
    update_dataset(os.path.join(augment_folder,'gen_evaluated.csv'),
                   os.path.join(augment_folder,'data.csv'),
                   target=args['target'],
                   threshold=args['cutoff_iterations'],
                   min_mol_wt=args['min_mol_wt'],
                   pairing_method=args['pairing_method'],
                   n_clusters=args['n_clusters'],
                   tan_threshold=args['tan_threshold']) # Reusing cutoff criteria defined for pairing algorithm
    
    # Return locations of folders
    return augment_folder

def run_iterations(args_file,save_dir,data_path,chemprop_path,num_iterations=1,constraint_file=None,solvent=None,starting_iteration=0):
    
    for iteration_num in range(starting_iteration,num_iterations):
        data_path = iterate_round(args_file,save_dir,data_path,chemprop_path,constraint_file,iteration_num=iteration_num,solvent=solvent)