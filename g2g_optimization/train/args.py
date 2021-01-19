import numpy as np
import re

def read_args(args_file):
    args = {}
    with open(args_file, 'r') as f:
        for line in f:
            split_line = line.split()
            if (len(split_line)>0): # Skip empty lines
                if (split_line[0][0]!='#'): # Skip lines with comments
                    if any(['#' in x for x in  split_line]):
                        comment = next((x for x in split_line if x[0]=='#'))
                        split_line= split_line[0:split_line.index(comment)] # Remove comments at end of line
                    if len(split_line)!=3:
                        raise Exception("Bad Line: {}".format(split_line))
                    key = split_line[0]
                    value = split_line[-1]
                    args[key] = value
                    
    # Correct data types:
    if 'cutoff' in list(args.keys()):
        args['cutoff'] = float(args['cutoff'])
    if 'sample_n' in list(args.keys()):
        args['sample_n'] = int(args['sample_n'])
    if 'remove_tails' in list(args.keys()):
        args['remove_tails'] = eval(args['remove_tails'])
    if 'batch_size' in list(args.keys()):
        args['batch_size'] = int(args['batch_size'])    
    if 'ncpu' in list(args.keys()):
        args['ncpu'] = int(args['ncpu'])    
    if 'load_dir' in list(args.keys()):
        if args['load_dir'] == 'None':
            args['load_dir'] = None
    if 'load_epoch' in list(args.keys()):
        args['load_epoch'] = int( args['load_epoch'])
    if 'hidden_size' in list(args.keys()):
        args['hidden_size'] = int( args['hidden_size'])    
    if 'embed_size' in list(args.keys()):
        args['embed_size'] = int( args['embed_size'])
    if 'latent_size' in list(args.keys()):
        args['latent_size'] = int( args['latent_size']) 
    if 'depthT' in list(args.keys()):
        args['depthT'] = int( args['depthT']) 
    if 'depthG' in list(args.keys()):
        args['depthG'] = int( args['depthG'])
    if 'diterT' in list(args.keys()):
        args['diterT'] = int( args['diterT'])
    if 'diterG' in list(args.keys()):
        args['diterG'] = int( args['diterG'])
    if 'dropout' in list(args.keys()):
        args['dropout'] = float(args['dropout'])
    if 'lr' in list(args.keys()):
        args['lr'] = float(args['lr'])
    if 'clip_norm' in list(args.keys()):
        args['clip_norm'] = float(args['clip_norm'])
    if 'beta' in list(args.keys()):
        args['beta'] = float(args['beta'])
    if 'epoch' in list(args.keys()):
        args['epoch'] = int(args['epoch'])
    if 'anneal_rate' in list(args.keys()):
        args['anneal_rate'] = float(args['anneal_rate'])      
    if 'print_iter' in list(args.keys()):
        args['print_iter'] = int(args['print_iter'])
    if 'save_iter' in list(args.keys()):
        args['save_iter'] = int(args['save_iter'])
    if 'min_mol_wt' in list(args.keys()):
        args['min_mol_wt'] = float(args['min_mol_wt'])
    if 'cutoff_iterations' in list(args.keys()):
        args['cutoff_iterations'] = float(args['cutoff_iterations'])
    if 'num_decode' in list(args.keys()):
        args['num_decode'] = int(args['num_decode'])        
    if 'sa_constraint' in list(args.keys()):
        args['sa_constraint'] = eval(args['sa_constraint'])
    if 'sa_cutoff' in list(args.keys()):
        args['sa_cutoff'] = float(args['sa_cutoff'])       
    if 'pairing_method' in list(args.keys()):
        args['pairing_method'] = args['pairing_method']
    if 'n_clusters' in list(args.keys()):
        args['n_clusters'] = int(args['n_clusters'])
    else:
        args['n_clusters'] = None
    if 'tan_threshold' in list(args.keys()):
        args['tan_threshold'] = float(args['tan_threshold'])  
    else:
        args['tan_threshold'] = None
    return args