import argparse
import rdkit
import os

from g2g_optimization.train.run_training import run_training



lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',required=True)
parser.add_argument('--save_dir',required=True)
parser.add_argument('--args_file',type=str, default=None) #Without an args file, many parameters will revert to default
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--chemprop_path', type=str, default='/data/rsg/chemistry/cbilod/chemprop/')
parser.add_argument('--constraint_file', type=str, default=None)

args = parser.parse_args()

if args.args_file == None:
    print('WARNING: You are running without an args_file')
    

run_training(args.data_path,args.save_dir,args.args_file,args.chemprop_path,args.constraint_file,args.input_file)