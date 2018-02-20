import argparse
import sys

sys.path.insert(0, 'utils')

from plot_helper import plot_progress


def plot(input_dir):
    plot_progress(input_folder=input_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots learning curves.')
    parser.add_argument('-d', '--dir', required=True,
                        help='Directory where the costs are saved')
    args = parser.parse_args()

    plot(input_dir=args.dir)