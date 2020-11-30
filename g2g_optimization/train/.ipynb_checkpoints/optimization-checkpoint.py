import os
import pickle
from functools import partial

from hyperopt import fmin, tpe, hp,Trials

from g2g_optimization.hgraph import common_atom_vocab
from g2g_optimization.train.args import read_args
from g2g_optimization.train.run_training import run_training
from g2g_optimization.train.evaluate_chemprop import evaluate_chemprop
from g2g_optimization.train.decode import decode

def opt_fn(hyperparams,
           data_path,
           test_set,
           save_dir,
           args_file,
           metric='percent_improved_mae',
           input_file=None,
           num_decode=20,
           seed=1):
    
    ## Get args:
    args = read_args(args_file)
    
    run_training(data_path,save_dir,args_file,input_file,hyperparams)
    
    vocab_file = os.path.join(save_dir,'inputs','vocab.txt')
    model_file = os.path.join(save_dir,'models','model.'+str(args['epoch']-1))
    
    if not os.path.isdir(os.path.join(save_dir,'eval')):
        os.mkdir(os.path.join(save_dir,'eval'))
        
    output_file = os.path.join(save_dir,'eval','decoded_mols.csv')
    stats_file = os.path.join(save_dir,'eval','stats.pkl')
    
    decode(test_set,vocab_file,model_file,output_file,args,
        atom_vocab=common_atom_vocab,
        num_decode=num_decode, ## Will not come from run input
        seed=seed,
        hyperparams=hyperparams)

    stats,_ = evaluate_chemprop(output_file,fold_path=args['fold_path'])

    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
        
    return stats[metric]

def objective(hyperparams,
               data_path,
               test_set,
               save_dir,
               args_file,
               metric='percent_improved_mae',
               input_file=None,
               num_decode=20,
               seed=1,
               maximize=True):
    
    folder_name = '_'.join([x+'_'+str(hyperparams[x]) for x in list(hyperparams.keys())])
    save_dir_parms = os.path.join(save_dir,folder_name)
    
    if not os.path.isdir(save_dir_parms):
        os.mkdir(save_dir_parms)
    
    performance = opt_fn(hyperparams,
                           data_path,
                           test_set,
                           save_dir_parms,
                           args_file,
                           metric,
                           input_file,
                           num_decode,
                           seed)
    if maximize:
        return -1.0*performance
    else:
        return performance
        

def optimization(data_path,
               test_set,
               save_dir,
               args_file,
               metric='percent_improved_mae',
               input_file=None,
               num_decode=20,
               seed=1,
               maximize=True):
    
    func = partial(objective, data_path=data_path,
                  test_set=test_set,
                  save_dir=save_dir,
                  args_file=args_file,
                  metric=metric,
                  input_file=input_file,
                  num_decode=num_decode,
                  seed=seed,
                  maximize=maximize)
        #All exepct hyperparams


    space={'latent_size':hp.choice('latent_size', range(4,16,4))} #to add hyperparams you need to modify run_training and decode
    trials = Trials()
    
    # Minimize objective
    fmin(func, space, trials=trials, algo=tpe.suggest, max_evals=10)
    
    
