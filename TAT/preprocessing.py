import argparse
import sys
import os

from .utils import preprocessing_cached_data
from .main import ROOT_DIR


def parse_args(argstring=None):
    parser = argparse.ArgumentParser('Converter script.')
    parser.add_argument('--datadir', type=str, default= os.path.join(ROOT_DIR, 'data/'), help='Dataset edge file name')
    parser.add_argument('--dataset', '-d', type=str, default='CollegeMsg', help='Dataset name')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio in the used data samples')
    parser.add_argument('--prop_depth', type=int, default=1, help='number of hops for one layer')
    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--max_sp', type=int, default=4, help='maximum distance to be encoded for shortest path feature (not used now)')

    try:
        if argstring is not None: args = parser.parse_args(argstring)
        else: args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    preprocessing_cached_data(dataset, args)
    


    