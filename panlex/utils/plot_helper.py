import csv
import pickle

import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, 'utils')
import strings


def plot_progress(logger, input_folder):
    logger.info('Plotting results...')
    output_folder = os.path.join(input_folder, strings.PLOT_FOLDER_NAME)
    os.makedirs(output_folder)
    # Plot learning curve
    loss_log_fn = os.path.join(input_folder, strings.TRAIN_FOLDER_NAME, strings.LOSS_LOG_FN)
    loss_plot_fn = os.path.join(output_folder, strings.LOSS_PLOT_FN)
    data_x = []
    data_y = []

    with open(loss_log_fn) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data_x.append(int(row[0]))
            data_y.append(float(row[1]))

    logger.info('max avg cos_sim: {0} at {1}'.format(max(data_y), data_y.index(max(data_y))))
    plt.plot(data_x, data_y, c='r')
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

    fig = plt.figure(figsize=(8,10))
    ax_1_2 = fig.add_subplot(211)
    ax_2_1 = fig.add_subplot(212)

    colors = ['b', 'g', 'y']
    prec_nbs = [1, 5, 10]
    prec_en_it = [[], [], []]
    prec_it_en = [[], [], []]
    prec_log_fn = os.path.join(input_folder, strings.TRAIN_FOLDER_NAME, strings.PREC_LOG_FN)
    prec_plot_fn = os.path.join(output_folder, strings.PREC_PLOT_FN)
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
    for i in range(nb):
        ax_1_2.plot(data_x, prec_en_it[i], c=colors[i], label='n={}'.format(prec_nbs[i]))
        logger.info('max prec {0}: {1} at {2}'
                    .format(prec_nbs[i], max(prec_en_it[i]), prec_en_it[i].index(max(prec_en_it[i]))))
    ax_1_2.legend()

    ax_2_1.set_title('Precision it-en')
    ax_2_1.set_xlabel('Epochs')
    ax_2_1.set_ylabel('Precision')
    ax_2_1.grid()
    for i in range(nb):
        ax_2_1.plot(data_x, prec_it_en[i], c=colors[i], label='n={}'.format(prec_nbs[i]))
        logger.info('max prec {0}: {1} at {2}'
                    .format(prec_nbs[i], max(prec_it_en[i]), prec_it_en[i].index(max(prec_it_en[i]))))
    ax_2_1.legend()

    fig.savefig(prec_plot_fn)