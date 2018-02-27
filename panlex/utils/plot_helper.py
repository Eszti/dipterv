import csv
import pickle

import matplotlib.pyplot as plt
import os
import sys


sys.path.insert(0, 'utils')
import strings
from io_helper import checkdir

def _log(logger, text):
    if logger is not None:
        logger.info(text)
    else:
        print(text)

def _get_data_xy(fn):
    data_x = []
    data_y = []
    with open(fn) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data_x.append(int(row[0]))
            data_y.append(float(row[1]))
    return data_x, data_y

# Todo: only for en-it
def plot_progress(input_folder, logger=None):
    _log(logger=logger, text='Plotting results...')
    output_folder = os.path.join(input_folder, strings.PLOT_OUTPUT_FOLDER_NAME)
    # Plot learning curve
    loss_u_log_fn = os.path.join(input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME, strings.LOSS_U_LOG_FN)
    loss_plot_fn = os.path.join(output_folder, strings.LOSS_PLOT_FN)
    checkdir(loss_plot_fn)

    data_x, data_y = _get_data_xy(loss_u_log_fn)
    _log(logger=logger, text='max avg cos_sim: {0} at {1}'.format(max(data_y), data_y.index(max(data_y))))
    plt.plot(data_x, data_y, c='r', label='univ')

    loss_l1_log_fn = os.path.join(input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME, strings.LOSS_L1_LOG_FN)
    if os.path.exists(loss_l1_log_fn):
        data_x, data_y = _get_data_xy(loss_l1_log_fn)
        plt.plot(data_x, data_y, c='g', label='l1')
    loss_l2_log_fn = os.path.join(input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME, strings.LOSS_L2_LOG_FN)
    if os.path.exists(loss_l2_log_fn):
        data_x, data_y = _get_data_xy(loss_l2_log_fn)
        plt.plot(data_x, data_y, c='b', label='l2')

    plt.title('Learning curve')
    plt.xlabel('Epochs')
    plt.ylabel('Avg sims')
    plt.grid()
    plt.savefig(loss_plot_fn)

    # Plot precision curves
    def append_precs(var, data):
        var[0].append(data[1][0])
        var[1].append(data[1][1])
        var[2].append(data[1][2])

    for eval_space in [strings.EVAL_SPACE_UNIV, strings.EVAL_SPACE_TARGET]:
        fig = plt.figure(figsize=(8,10))
        ax_1_2 = fig.add_subplot(211)
        ax_2_1 = fig.add_subplot(212)

        prec_log_fn = os.path.join(
            input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME, strings.PREC_LOG_FN, '_{}'.format(eval_space))
        if os.path.exists(prec_log_fn):
            logger.info('{} space'.format(eval_space))
            colors = ['b', 'g', 'y']
            prec_nbs = [1, 5, 10]
            prec_en_it = [[], [], []]
            prec_it_en = [[], [], []]
            prec_plot_fn = os.path.join(output_folder, '{0}_{1}.png'.format(strings.PREC_PLOT_FN, eval_space))
            with open(prec_log_fn, 'rb') as picklefile:
                precs = pickle.load(picklefile)
            for i, ls in enumerate(precs):
                en_it = ls[0]
                it_en = ls[1]
                append_precs(prec_en_it, en_it)
                append_precs(prec_it_en, it_en)

            nb = len(prec_en_it)
            ax_1_2.set_title('Precision en-it')
            ax_1_2.set_xlabel('Epochs')
            ax_1_2.set_ylabel('Precision')
            ax_1_2.grid()
            logger.info('en-it')
            for i in range(nb):
                ax_1_2.plot(data_x, prec_en_it[i], c=colors[i], label='n={}'.format(prec_nbs[i]))
                _log(logger=logger, text='max prec {0}: {1} at {2}'
                     .format(prec_nbs[i], max(prec_en_it[i]), prec_en_it[i].index(max(prec_en_it[i]))))
            ax_1_2.legend()

            ax_2_1.set_title('Precision it-en')
            ax_2_1.set_xlabel('Epochs')
            ax_2_1.set_ylabel('Precision')
            ax_2_1.grid()
            logger.info('it-en')
            for i in range(nb):
                ax_2_1.plot(data_x, prec_it_en[i], c=colors[i], label='n={}'.format(prec_nbs[i]))
                _log(logger=logger, text='max prec {0}: {1} at {2}'
                     .format(prec_nbs[i], max(prec_it_en[i]), prec_it_en[i].index(max(prec_it_en[i]))))

            ax_2_1.legend()

            fig.savefig(prec_plot_fn)