import sys
from g2g_optimization.hgraph import *
from rdkit import Chem
from multiprocessing import Pool
import pandas as pd

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

  
def get_vocab(mol_file,vocab_file):

    #data = [mol for line in sys.stdin for mol in line.split()[:2]]
    #data = list(set(data))
    data = list(set([x[0] for x in pd.read_csv(mol_file,header=None).values]))

    ncpu = 15
    batch_size = len(data) // ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(ncpu)
    vocab_list = pool.map(process, batches)
        
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    f = open(vocab_file, 'w')

    for x,y in sorted(vocab):
        f.write(x+' '+y+'\n')
