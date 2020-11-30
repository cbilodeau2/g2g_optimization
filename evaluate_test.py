import rdkit
import argparse

from g2g_optimization.train.decode import decode
from g2g_optimization.train.args import read_args
from g2g_optimization.hgraph import common_atom_vocab


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test',required=True)
parser.add_argument('--vocab',required=True)
parser.add_argument('--model',required=True)
parser.add_argument('--output_file',required=True)
parser.add_argument('--args_file',type=str, default=None) #Without an args file, many parameters will revert to default
parser.add_argument('--num_decode',type=int, default=20)
parser.add_argument('--seed',type=int, default=1)

args = parser.parse_args()

if args.args_file == None:
    print('WARNING: You are running without an args_file')
    args_file = {}
else:
    args_file = read_args(args.args_file)
    
decode(args.test,args.vocab,args.model,args.output_file,args_file,
        atom_vocab=common_atom_vocab,
        num_decode=args.num_decode, ## Will not come from run input
        seed=args.seed)
