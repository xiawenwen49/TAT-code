import argparse
import numpy as np
import sys
import torch
import os
import time
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from . import log
from . import utils
from .model import get_model
from .train import train_model

import torch
torch.autograd.set_detect_anomaly(True)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser('Temporal GNNs.')
    
    # general settings
    parser.add_argument('--root_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='Root directory' )
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(ROOT_DIR, 'checkpoint/'), help='Root directory' )
    parser.add_argument('--datadir', type=str, default= os.path.join(ROOT_DIR, 'data/'), help='Dataset edge file name')
    parser.add_argument('--dataset', type=str, default='CollegeMsg', choices=['CollegeMsg', 'emailEuCoreTemporal', 'SMS-A', 'facebook-wall'], help='Dataset edge file name')
    parser.add_argument('--force_cache', default=False, action='store_true', help='use cahced dataset if exists')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu, others: gpu index')
    parser.add_argument('--set_indice_length', type=int, default=3, help='number of nodes in set_indice, for TAT Model')

    # dataset 
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--data_usage', type=float, default=0.6, help='ratio of used data for all data samples')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio in the used data samples')
    parser.add_argument('--parallel', default=False, action='store_true', help='parallelly generate subgraphs')

    # model training
    parser.add_argument('--model', type=str, default='TAT', choices=['TAT', 'DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT', 'TAGCN'], help='model name')
    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='for leakey relu function')
    parser.add_argument('--use_attention', type=str2bool, default=True, help='use attention or not in TAT model')
    
    parser.add_argument('--epoch', type=int, default=50, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')
    parser.add_argument('--lr', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-3, help='l2 regularization')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer (string)')
    parser.add_argument('--metric', type=str, default='acc', help='evaluation metric')
    parser.add_argument('--perm_loss', type=float, default=0.1, help='weight of permutation loss') 

    # important features
    parser.add_argument('--in_features', type=int, default=9, help='initial input features of nodes')
    parser.add_argument('--out_features', type=int, default=6, help='number of target classes')
    parser.add_argument('--prop_depth', type=int, default=1, help='number of hops for one layer')
    parser.add_argument('--max_sp', type=int, default=4, help='maximum distance to be encoded for shortest path feature (not used now)')
    parser.add_argument('--sp_feature_type', type=str, choices=['sp', 'rw'], default='sp', help='spatial features type, shortest path, or random landing probabilities')

    parser.add_argument('--time_encoder_type', type=str, default='tat', choices=['tat', 'harmonic', 'empty'], help='time encoder type')
    parser.add_argument('--time_encoder_maxt', type=float, default=3e6, help='time encoder maxt')
    parser.add_argument('--time_encoder_rows', type=int, default=int(1e6), help='time encoder rows')
    parser.add_argument('--time_encoder_dimension', type=int, default=64, help='time encoding dimension')
    parser.add_argument('--time_encoder_discrete', type=str, default='uniform', choices=['uniform', 'log'], help='discrete type')
    parser.add_argument('--time_encoder_deltas', type=float, default=0.5, help='scale of mean time interval for discretization')

    # logging and debug
    parser.add_argument('--log_dir', type=str, default=os.path.join(ROOT_DIR, 'log/'), help='log directory')
    parser.add_argument('--save_log', default=False, action='store_true', help='save console log into log file')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--desc', type=str, default='description_string', help='a string description for an experiment')
    parser.add_argument('--time_str', type=str, default=time.strftime('%Y_%m_%d_%H_%M_%S'), help='execution time')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

def main():
    # arg and log settings
    args = parse_args()
    sys_argv = sys.argv
    logger = log.set_up_log(args, sys_argv)

    # read in dataset
    G = utils.read_file(args.dataset, args)

    # load dataset
    dataloaders, in_dim = utils.load_dataloaders(G, args)

    if args.in_features != in_dim:
        logger.info("assiged feature dim is not comparable with data dim, force updated {}.".format(in_dim))
        args.in_features = in_dim
    
    mod = get_model(args, logger)

    results = train_model(mod, dataloaders, args, logger)

if __name__ == "__main__":
    main()
