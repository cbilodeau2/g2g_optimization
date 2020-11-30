import os
from shutil import copyfile
import time

from g2g_optimization.train.get_vocab import get_vocab
from g2g_optimization.hgraph.pairing import generate_pairs
from g2g_optimization.hgraph import common_atom_vocab
from g2g_optimization.train.preprocess import generate_tensors
from g2g_optimization.train.gnn_train import gnn_train
from g2g_optimization.train.args import read_args

def run_training(data_path='data/solvation_open',save_dir='checkpoints',args_file=None,chemprop_path='/data/rsg/chemistry/cbilod/chemprop/',constraint_file=None,input_file=None,hyperparams=None):

    args = {}
    if args_file!=None:
        args = read_args(args_file)
    
    if input_file==None:
        input_file = os.path.join(data_path,'data.csv')
        
    ## Hyperparameters:-------
    
    if hyperparams !=None:
        if 'latent_size' in list(hyperparams.keys()):
            args['latent_size'] = hyperparams['latent_size']
    
    #--------------------------
        
        
    if not os.path.isdir(os.path.join(save_dir,'inputs')):
        os.mkdir(os.path.join(save_dir,'inputs'))
    input_dir = os.path.join(save_dir,'inputs')
    
    mol_file = os.path.join(input_dir,'mols.txt')
    vocab_file = os.path.join(input_dir,'vocab.txt')
    train_file = os.path.join(input_dir,'train_pairs.txt')

    if not os.path.isdir(os.path.join(save_dir,'tensors')):
        os.mkdir(os.path.join(save_dir,'tensors'))
    tensor_dir = os.path.join(save_dir,'tensors')
    
    if not os.path.isdir(os.path.join(save_dir,'models')):
        os.mkdir(os.path.join(save_dir,'models'))
    model_dir = os.path.join(save_dir,'models')

    log_file = os.path.join(save_dir,'run.log')
    f = open(log_file, 'w')
    f.write('Arguments:\n')
    f.write(str(args)+'\n')
    f.write('\n')
    
    f.write('Starting Pair Generation \n')
    start = time.time()
    generate_pairs(input_file, train_file, mol_file,args,chemprop_path,constraint_file)
    end = time.time()
    f.write('Ending Pair Generation \n')
    f.write('Time Elapsed: '+str(end-start)+'\n')
    f.write('\n')
    
    f.write('Starting Vocab \n')
    start = time.time()
    get_vocab(mol_file,vocab_file)
    end = time.time()
    f.write('Ending Vocab \n')
    f.write('Time Elapsed: '+str(end-start)+'\n')
    f.write('\n')
    
    f.write('Starting Preprocessing \n')
    start = time.time()
    generate_tensors(train_file,vocab_file,tensor_dir,args)
    end = time.time()
    f.write('Ending Preprocessing \n')
    f.write('Time Elapsed: '+str(end-start) + '\n')
    f.write('\n')
    
    f.write('Starting Model Training \n')
    start = time.time()
    gnn_train(tensor_dir,vocab_file,model_dir,args)
    end = time.time()
    f.write('Ending Model Training \n')
    f.write('Time Elapsed: '+str(end-start)+ '\n')
    f.write('\n')
    
    if args_file!=None:
        copyfile(args_file,os.path.join(save_dir,'input.dat'))
    
    
    
    
