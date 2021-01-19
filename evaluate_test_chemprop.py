import rdkit
import argparse
import pickle
import os

from g2g_optimization.train.decode impodecode
from g2g_optimization.train.args import read_args
from g2g_optimization.hgraph import common_atom_vocab
from g2g_optimization.train.evaluate_chemprop import evaluate_chemprop

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test',required=True)
parser.add_argument('--model',required=True)
parser.add_argument('--vocab',type=str,default=None)
parser.add_argument('--output_file',type=str,default=None)
parser.add_argument('--stats_file',type=str,default=None)
parser.add_argument('--checkpoint_path',type=str,default=None)
parser.add_argument('--fold_path',default='predictors/chemprop_aqsol/')
parser.add_argument('--args_file',type=str, default=None) #Without an args file, many parameters will revert to default
parser.add_argument('--num_decode',type=int, default=20)
parser.add_argument('--seed',type=int, default=1)
parser.add_argument('--chemprop_path',type=str, default='/data/rsg/chemistry/cbilod/chemprop/')
parser.add_argument('--solvent',type=str, default=None)

args = parser.parse_args()

    
if args.checkpoint_path !=None:
    args.vocab = os.path.join(args.checkpoint_path,'inputs','vocab.txt')
    args.model = os.path.join(args.checkpoint_path,'models',args.model)
    if not os.path.isdir(os.path.join(args.checkpoint_path,'eval')):
        os.mkdir(os.path.join(args.checkpoint_path,'eval'))
    args.output_file = os.path.join(args.checkpoint_path,'eval','decoded_mols.csv')
    args.stats_file = os.path.join(args.checkpoint_path,'eval','stats.pkl')
    args.args_file = os.path.join(args.checkpoint_path,'input.dat')

if args.args_file == None:
    print('WARNING: You are running without an args_file')
    args_file = {}
else:
    args_file = read_args(args.args_file)
    
decode(args.test,args.vocab,args.model,args.output_file,args_file,
        atom_vocab=common_atom_vocab,
        num_decode=args.num_decode, ## Will not come from run input
        seed=args.seed)

if args.solvent == None:
    stats,_ = evaluate_chemprop(args.output_file,fold_path=args.fold_path,chemprop_path=args.chemprop_path)
else:
    stats,_ = evaluate_chemprop_sol(out_file,solvent=args.solvent,fold_path=args.fold_path,chemprop_path=args.chemprop_path)
    
with open(args.stats_file, 'wb') as f:
    pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)


