import csv
import pickle

import matplotlib.pyplot as plt
import os
import sys


sys.path.insert(0, 'utils')
import strings
from io_helper import checkdir, tsv_into_arrays, get_matching_files


def _log(logger, text):
    if logger is not None:
        logger.info(text)
    else:
        print(text)

def plot_progress(input_folder, logger=None, ymin=0.7):
    _log(logger=logger, text='Plotting results...')
    output_folder = os.path.join(input_folder, strings.PLOT_OUTPUT_FOLDER_NAME)
    delim = '\t'

    # Plot learning curve
    sim_log_fn = os.path.join(input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME, strings.SIM_LOG_FN)
    sim_plot_fn = os.path.join(output_folder, strings.SIM_PLOT_FN)
    checkdir(sim_plot_fn)
    plt.yscale('linear')
    # Get train loss data
    data = tsv_into_arrays(sim_log_fn, delim=delim)
    max_val = max(data[1])
    max_idx = data[1].index(max_val)
    _log(logger=logger, text='max avg cos_sim (train): {0} at {1}'.format(max_val, int(data[0][max_idx])))
    plt.plot(data[0], data[1], c='r', label='train')
    # Get valid loss data
    valid_sim_log_fn = os.path.join(input_folder, strings.VALID_OUTPUT_FOLDER_NAME, strings.SIM_LOG_FN)
    if os.path.exists(valid_sim_log_fn):
        data = tsv_into_arrays(valid_sim_log_fn, delim=delim)
        max_val = max(data[1])
        max_idx = data[1].index(max_val)
        _log(logger=logger, text='max avg cos_sim (valid): {0} at {1}'.format(max_val, int(data[0][max_idx])))
        plt.plot(data[0], data[1], c='g', label='valid')
    # Plot
    plt.legend()
    plt.title('Learning curve')
    plt.xlabel('Epochs')
    plt.ylabel('Avg sims')
    plt.ylim(ymin=ymin)
    plt.grid()
    plt.savefig(sim_plot_fn)
    plt.close()

    # Plot precision curves
    COLORS_RGB = [
        (228, 26, 28), (55, 126, 184), (77, 175, 74),
        (152, 78, 163), (255, 127, 0), (255, 255, 51),
        (166, 86, 40), (247, 129, 191), (153, 153, 153)
    ]
    colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in COLORS_RGB]

    valid_dir = os.path.join(input_folder, strings.VALID_OUTPUT_FOLDER_NAME)
    if os.path.exists(valid_dir):
        prec_files = get_matching_files(valid_dir, '{}*'.format(strings.PREC_LOG_FN))

        for fn in prec_files:
            parts = fn.split('_')
            l1, l2 = parts[-2], parts[-1]
            input_fn = os.path.join(input_folder, strings.VALID_OUTPUT_FOLDER_NAME, fn)
            data = tsv_into_arrays(input_fn, delim=delim)
            nb_precs = len(data) - 1
            prec_plot_fn = os.path.join(output_folder, '{0}_{1}_{2}.png'.format(strings.PREC_PLOT_FN, l1, l2))
            _log(logger, '{0}-{1}'.format(l1, l2))
            for i in range(nb_precs):
                max_val = max(data[i+1][1:])
                max_idx = data[i+1].index(max_val)
                _log(logger=logger, text='max prec {0}: {1} at {2}'
                     .format(int(data[i+1][0]), max_val, int(data[0][max_idx])))
                plt.plot(data[0][1:], data[i+1][1:], c=colors[i], label='{}'.format(data[i+1][0]))
            plt.legend()
            plt.title('Precision {0}-{1}'.format(l1, l2))
            plt.xlabel('Epochs')
            plt.ylabel('Precision')
            plt.grid()
            plt.savefig(prec_plot_fn)
            plt.close()