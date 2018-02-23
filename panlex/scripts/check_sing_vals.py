import argparse
import pickle
import sys

import os

sys.path.insert(0, 'utils')


def get_sing_vals(output_folder, nbs, limit, logger=False):
    print(nbs)
    for i, nb in enumerate(nbs):
        T_fn = os.path.join(output_folder, 'train_mod', 'T_{}.pickle'.format(nb))
        with open(T_fn, 'rb') as f:
            T = pickle.load(f)
        T_en = T[0]
        log('eng', logger)
        get_smalls(T_en, limit, nb, logger)
        T_it = T[1]
        print('ita', logger)
        get_smalls(T_it, limit, nb, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots learning curves.')
    parser.add_argument('-d', '--dir', required=True,
                        help='Directory where the costs are saved')
    parser.add_argument('-l', '--limit', type=float, required=False, default=0.1,
                        help='Count singular values smaller than limit. Default = 0.1')
    parser.add_argument('-n', '--numbers', type=int, required=True, nargs='+',
                        help='Epoch numbers to check')
    args = parser.parse_args()

    get_sing_vals(output_folder=args.dir, limit=args.limit, nbs=args.numbers)