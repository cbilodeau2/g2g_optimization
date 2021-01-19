import argparse
import rdkit
import os

from g2g_optimization.train.run_iterations import run_iterations


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',required=True)
parser.add_argument('--save_dir',required=True)
parser.add_argument('--args_file',type=str, default=None) #Without an args file, many parameters will revert to default
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--n_iterations', type=int, default=1)
parser.add_argument('--chemprop_path',type=str, default='/data/rsg/chemistry/cbilod/chemprop/')
parser.add_argument('--constraint_file',type=str, default=None)
parser.add_argument('--solvent',type=str, default=None)
parser.add_argument('--starting_iteration',type=int, default=0)

args = parser.parse_args()

run_iterations(args.args_file,args.save_dir,args.data_path,args.chemprop_path,num_iterations=args.n_iterations,constraint_file=args.constraint_file,solvent=args.solvent,starting_iteration=args.starting_iteration)







