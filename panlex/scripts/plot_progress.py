import argparse
import sys

sys.path.insert(0, 'utils')

from plot_helper import plot_progress


def plot(input_dir, ymin):
    if ymin is None:
        plot_progress(input_folder=input_dir)
    else:
        plot_progress(input_folder=input_dir, ymin=ymin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots learning curves.')
    parser.add_argument('-d', '--dir', required=True,
                        help='Directory where the costs are saved')
    parser.add_argument('-ymin', '--ymin', required=False, type=float,
                        help='Y axis minimum range')
    args = parser.parse_args()

    plot(input_dir=args.dir, ymin=args.ymin)