import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import pandas as pd

import math, random, sys
import numpy as np
import io

from g2g_optimization.hgraph.hgnn import HierVGNN
from g2g_optimization.hgraph.pairing import check_atoms
from g2g_optimization.hgraph.vocab import PairVocab,common_atom_vocab
from g2g_optimization.hgraph.dataset import MolEnumRootDataset



def decode(test,vocab,model_path,output_file,args,
           atom_vocab=common_atom_vocab,
           num_decode=20, ## Will not come from run input
           seed=1, ## Will not come from run input
           hyperparams=None,
           rnn_type='LSTM',
           hidden_size=270,
           embed_size=270,
           batch_size=32,
           latent_size=4,
           depthT=20,
           depthG=20,
           diterT=1,
           diterG=3,
           dropout=0.0):
    
    ## Hyperparameters:-------

    if hyperparams !=None:
        if 'latent_size' in list(hyperparams.keys()):
            args['latent_size'] = hyperparams['latent_size']           
    
    #--------------------------
    
    
    # Read from args dict:
    if 'rnn_type' in list(args.keys()):
        rnn_type = args['rnn_type']     
    if 'hidden_size' in list(args.keys()):
        hidden_size = args['hidden_size']     
    if 'embed_size' in list(args.keys()):
        embed_size = args['embed_size']     
    if 'batch_size' in list(args.keys()):
        batch_size = args['batch_size']  
    if 'latent_size' in list(args.keys()):
        latent_size = args['latent_size']         
    if 'depthT' in list(args.keys()):
        depthT = args['depthT']        
    if 'depthG' in list(args.keys()):
        depthG = args['depthG'] 
    if 'diterT' in list(args.keys()):
        diterT = args['diterT']
    if 'diterG' in list(args.keys()):
        diterG = args['diterG']          
    if 'dropout' in list(args.keys()):
        dropout = args['dropout']

    
    
    test = pd.DataFrame([line.strip("\r\n ") for line in open(test)])
    
    start_length = len(test)
    test = test[test[0].apply(lambda x: not '.' in x)] # drop dots
    test = test[test[0].apply(check_atoms)] # Remove unusual atoms
    end_length = len(test)
    print('{} test molecules skipped due to incompatibility'.format(start_length-end_length))
    test = [x[0] for x in test.values]
    
    
    vocab = [x.strip("\r\n ").split() for x in open(vocab)] 
    vocab = PairVocab(vocab)

    model = HierVGNN(vocab=vocab,
                     atom_vocab=atom_vocab,
                     rnn_type=rnn_type,
                     hidden_size=hidden_size,
                     embed_size=embed_size,
                     batch_size=batch_size,
                     latent_size=latent_size,
                     depthT=depthT,
                     depthG=depthG,
                     diterT=diterT,
                     diterG=diterG,
                     dropout=dropout).cuda() 
    

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer))
    model.eval()

    dataset = MolEnumRootDataset(test, vocab, atom_vocab)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

    torch.manual_seed(seed)
    random.seed(seed)

    enum_root=True
    greedy=True
    
    g = open(output_file, 'w')
    with torch.no_grad():
        for i,batch in enumerate(loader):
            smiles = test[i]
            if batch is None:
                for k in range(num_decode):
                    #print(smiles, smiles)
                    g.write(smiles+' '+smiles+'\n')
            else:
                new_mols = model.translate(batch[1], num_decode, enum_root, greedy)
                for k in range(num_decode):
                    g.write(smiles+' '+new_mols[k]+'\n')
                    #print(smiles, new_mols[k]) 

