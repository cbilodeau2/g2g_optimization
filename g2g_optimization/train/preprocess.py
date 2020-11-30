from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy

from g2g_optimization.hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x

def generate_tensors(train_file,vocab_file,tensor_path, args,batch_size=32,ncpu=8):

    # Read from args dict:
    if 'batch_size' in list(args.keys()):
        batch_size = args['batch_size']
    if 'ncpu' in list(args.keys()):
        ncpu = args['ncpu']  

    with open(vocab_file) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    vocab_file = PairVocab(vocab, cuda=False)

    pool = Pool(ncpu) 
    random.seed(1)

    with open(train_file) as f:
        data = [line.strip("\r\n ").split()[:2] for line in f]

    random.shuffle(data)

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    func = partial(tensorize_pair, vocab = vocab_file)
    all_data = pool.map(func, batches)
    num_splits = max(len(all_data) // 1000, 1)

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open(tensor_path+'/tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

