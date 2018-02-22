import argparse
import sys
import numpy as np
import pickle
import os

sys.path.insert(0, 'utils')

from plot_helper import plot_progress


def get_sing_vals(output_folder, nbs, limit, print_s=False):
    print(nbs)
    def get_smalls(T):
        U, s, V = np.linalg.svd(T, full_matrices=True, compute_uv=True)
        small = np.where(s < limit)
        print('T_{0}\t<{1} : {2}'.format(nb, limit, len(small[0])))
        if print_s:
            print(s)
    for i, nb in enumerate(nbs):
        T_fn = os.path.join(output_folder, 'train_mod', 'T_{}.pickle'.format(nb))
        with open(T_fn, 'rb') as f:
            T = pickle.load(f)
        T_en = T[0]
        print('eng')
        get_smalls(T_en)
        T_it = T[1]
        print('ita')
        get_smalls(T_it)

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