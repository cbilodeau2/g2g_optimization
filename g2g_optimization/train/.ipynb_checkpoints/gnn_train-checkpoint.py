import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
import os

from g2g_optimization.hgraph import *
import rdkit

def gnn_train(tensor_path,vocab_file,model_dir,args,
              load_dir=None,
              load_epoch=-1,
              rnn_type='LSTM',
              hidden_size=270,
              embed_size=270,
              batch_size=32,
              latent_size=4,
              depthT=20,
              depthG=20,
              diterT=1,
              diterG=3,
              dropout=0.0,
              lr=1e-3,
              clip_norm=20.0,
              beta=0.3,
              epoch=12,
              anneal_rate=0.9,
              print_iter=50,
              save_iter=-1,
              atom_vocab=common_atom_vocab):
    
    # Read from args dict:
    if 'load_dir' in list(args.keys()):
        load_dir = args['load_dir']
    if 'load_epoch' in list(args.keys()):
        load_epoch = args['load_epoch']  
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
    if 'lr' in list(args.keys()):
        lr = args['lr']        
    if 'beta' in list(args.keys()):
        beta = args['beta']
    if 'epoch' in list(args.keys()):
        epoch = args['epoch'] 
    if 'anneal_rate' in list(args.keys()):
        anneal_rate = args['anneal_rate'] 
    if 'print_iter' in list(args.keys()):
        print_iter = args['print_iter']
    if 'save_iter' in list(args.keys()):
        save_iter = args['save_iter']     

    
    vocab = [x.strip("\r\n ").split() for x in open(vocab_file)] 
    vocab_file = PairVocab(vocab)

    model = HierVGNN(vocab=vocab_file,
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


    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if load_epoch >= 0:
        model.load_state_dict(torch.load(load_dir + "/model." + str(load_epoch)))

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = 0
    beta = beta
    meters = np.zeros(6)

    for epoch in range(load_epoch + 1, epoch):
        dataset = DataFolder(tensor_path, batch_size)

        for batch in dataset:
            total_step += 1
            batch = batch + (beta,)
            model.zero_grad()
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])

            if total_step % print_iter == 0:
                meters /= print_iter
                print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0

            if save_iter >= 0 and total_step % save_iter == 0:
                n_iter = total_step // save_iter - 1
                torch.save(model.state_dict(), model_dir + "/model." + str(n_iter))
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

        del dataset
        if save_iter == -1:
            torch.save(model.state_dict(), model_dir + "/model." + str(epoch))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])
