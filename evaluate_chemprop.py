import rdkit
import argparse
import pickle
import os

from g2g_optimization.train.decode import decode
from g2g_optimization.train.args import read_args
from g2g_optimization.hgraph import common_atom_vocab
from g2g_optimization.train.evaluate_chemprop import evaluate_chemprop,evaluate_chemprop_sol
from g2g_optimization.train.generate_tests import generate_test_sets

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--data_path',required=True)
parser.add_argument('--checkpoint_path',type=str,default=None)
parser.add_argument('--vocab',type=str,default=None)
parser.add_argument('--target',type=str,default='Solubility')
parser.add_argument('--fold_path',default='/predictors/chemprop_aqsol/')
parser.add_argument('--args_file',type=str, default=None) #Without an args file, many parameters will revert to default
parser.add_argument('--num_decode',type=int, default=20)
parser.add_argument('--seed',type=int, default=1)
parser.add_argument('--chemprop_path',type=str, default='/data/rsg/chemistry/cbilod/chemprop/')
parser.add_argument('--solvent',type=str, default=None)


args = parser.parse_args()

# You can just input the checkpoint path, this populates everything else:
    
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
    
print(args_file['target'])
# Generate test sets:
generate_test_sets(os.path.join(args.data_path,'data.csv'),
                   os.path.join(args.checkpoint_path,'eval'),
                   target=args_file['target'])#args.target)

for test_set in ['bottom','med_bottom','medium','med_top','top']:
    test_file = os.path.join(args.checkpoint_path,'eval',test_set+'.txt')
    out_file = os.path.join(args.checkpoint_path,'eval','out_'+test_set+'.txt')
    out_stats_file = os.path.join(args.checkpoint_path,'eval','stats_'+test_set+'.pkl')
    decode(test_file,args.vocab,args.model,out_file,args_file,
            atom_vocab=common_atom_vocab,
            num_decode=args.num_decode, ## Will not come from run input
            seed=args.seed)

    if args.solvent == None:
        stats,_ = evaluate_chemprop(out_file,fold_path=args.fold_path,chemprop_path=args.chemprop_path)
    else:
        stats,_ = evaluate_chemprop_sol(out_file,solvent=args.solvent,fold_path=args.fold_path,chemprop_path=args.chemprop_path)

    with open(out_stats_file, 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)