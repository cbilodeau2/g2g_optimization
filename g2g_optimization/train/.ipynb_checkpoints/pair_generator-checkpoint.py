import numpy as np
import sys
import os
from chemprop.data.data import MoleculeDatapoint
from chemprop.data.data import MoleculeDataset
from chemprop.data.scaffold import scaffold_split
from chemprop.data.scaffold import *
import argparse
import pandas as pd
import time
from rdkit import Chem
from hgraph.pairing import *

from hgraph import common_atom_vocab


            
if __name__ == "__main__":
    
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True)
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--molfile', required=True)
    parser.add_argument('--target', type=str,default='Solubility')
    parser.add_argument('--cutoff', type=float, default=0.78*2)
    parser.add_argument('--sample', type=int, default=20)
    parser.add_argument('--remove_tails', type=bool, default=True)
    
    args = parser.parse_args()
    
    generate_pairs(args.infile,
                   args.outfile,
                   args.molfile,
                   args.target,
                   args.cutoff,
                   args.sample,
                   args.remove_tails)
    
    end = time.time()
    
    print('Completed. Time Elapsed:{}'.format(end-start))