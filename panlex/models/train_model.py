import sys

import copy
import os

sys.path.insert(0, 'utils')

from io_helper import save_pickle, list_to_csv
from math_helper import calc_precision


import strings
from base.loggable import Loggable
import tensorflow as tf
import numpy as np

class TrainMModel(Loggable):
    def __init__(self, train_config, data_model_wrapper, language_config, output_dir):
        Loggable.__init__(self)
        self.train_config = train_config
        self.data_model_wrapper = data_model_wrapper
        self.langs = language_config.langs
        self.logger.info('Language order: {0}'.format([(i, l) for i, l in enumerate(self.langs)]))
        self.dim = data_model_wrapper.dim
        self.train_data_model = self.data_model_wrapper.data_models[strings.TRAIN]
        self.valid_data_model = self.data_model_wrapper.data_models[strings.VALID]
        # Getting embeddings
        self.train_embeddings = self.data_model_wrapper.training_embeddings
        self.embeddings = self.data_model_wrapper.embedding_model.embeddings
        self.output_dir = os.path.join(output_dir, strings.TRAIN_FOLDER_NAME)

    def train(self):
        nb_langs = len(self.langs)

        # Init graphs
        graph = tf.Graph()
        with graph.as_default():
            # TF variables
            # Placeholder for 2 words
            tf_w1 = tf.placeholder(tf.float32, shape=[None, self.dim])
            tf_w2 = tf.placeholder(tf.float32, shape=[None, self.dim])
            # Placeholder for indexing the T matrix
            tf_idx_l1 = tf.placeholder(tf.int32)
            tf_idx_l2 = tf.placeholder(tf.int32)
            # Translation matrices
            tf_T = tf.Variable(tf.truncated_normal([nb_langs, self.dim, self.dim]))

            # SVD reguralization
            tf_s1, tf_U1, tf_V1 = tf.svd(tf_T[tf_idx_l1], full_matrices=True, compute_uv=True)
            updated_1 = tf.assign(tf_T[tf_idx_l1], tf.matmul(tf_U1, tf_V1))
            tf_s2, tf_U2, tf_V2 = tf.svd(tf_T[tf_idx_l2], full_matrices=True, compute_uv=True)
            updated_2 = tf.assign(tf_T[tf_idx_l2], tf.matmul(tf_U2, tf_V2))

            # Loss
            tf_T1 = tf.matmul(tf_w1, tf_T[tf_idx_l1])
            tf_T2 = tf.matmul(tf_w2, tf_T[tf_idx_l2])
            tf_T1_n = tf.nn.l2_normalize(tf_T1, dim=1)
            tf_T2_n = tf.nn.l2_normalize(tf_T2, dim=1)
            loss = tf.matmul(tf_T1_n, tf.transpose(tf_T2_n))
            loss = -loss

            # Applying optimizer, Todo: try different optimizers!!
            # https://www.tensorflow.org/api_guides/python/train#Optimizers
            optimizer = tf.train.AdagradOptimizer(self.train_config.lr_base).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            j = 0
            lc_arr = []
            precs_arr = []
            for i in range(self.train_config.epochs):
                loss_arr = []
                for ((l1, l2), wp_l) in self.train_data_model.word_pairs_dict.items():
                    loss_arr_l = []
                    idx_l1 = self.langs.index(l1)
                    idx_l2 = self.langs.index(l2)
                    k = 0
                    for (w1, w2) in wp_l:
                        emb1 = self.train_embeddings[l1][w1].reshape((1, 300))
                        emb2 = self.train_embeddings[l2][w2].reshape((1, 300))
                        if (self.train_config.svd_mode == 1 and i % self.train_config.svd_f == 0) or \
                                (self.train_config.svd_mode == 2 and j == 0):
                            _, l, _, _, T = session.run([optimizer, loss, updated_1, updated_2, tf_T],
                                                        feed_dict={tf_w1: emb1,
                                                                   tf_w2: emb2,
                                                                   tf_idx_l1: idx_l1,
                                                                   tf_idx_l2: idx_l2})
                        else:
                            _, l, T = session.run([optimizer, loss, tf_T],
                                                  feed_dict={tf_w1: emb1,
                                                             tf_w2: emb2,
                                                             tf_idx_l1: idx_l1,
                                                             tf_idx_l2: idx_l2})
                        j += 1
                        k += 1
                        loss_arr.append(-l[0][0])
                        loss_arr_l.append(-l[0][0])
                        if self.train_config.iters is not None and j == self.train_config.iters:
                            break
                    if self.train_config.iters is not None and j == self.train_config.iters:
                        break

                # Monitoring for learning curve
                loss_np_arr = np.asarray(loss_arr)
                loss_epoch_avg = np.average(loss_np_arr)
                self.logger.info('epoch:\t{0}\tavg sims: {1}'.format(i, loss_epoch_avg))
                lc_arr.append([i, loss_epoch_avg])

                if self.train_config.do_prec_calc:
                    # Calculate precision
                    e_prec_l = []
                    for ((l1, l2), _) in self.train_data_model.word_pairs_dict.items():
                        self.logger.info('Calculating precision for {0}-{1}'.format(l1, l2))
                        # Get translations matrices
                        idx_l1 = self.langs.index(l1)
                        idx_l2 = self.langs.index(l2)
                        T1 = T[idx_l1]
                        T2 = T[idx_l2]
                        precs_1 = 0.0
                        precs_2 = 0.0
                        if self.train_config.prec_calc_strat == 0 or self.train_config.prec_calc_strat == 1:
                            m1_tr = copy.deepcopy(self.train_data_model.filtered_models[(l1, l2)])
                            m2_tr = copy.deepcopy(self.train_data_model.filtered_models[(l2, l1)])
                            m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
                            m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
                            if self.train_config.prec_calc_strat == 0:
                                # Prec l1 - l2
                                precs_1 = calc_precision(self.train_config.precs_to_calc, m1_tr, m2_tr,
                                                         self.train_data_model.dictionaries[(l1, l2)],
                                                         self.logger)
                                # Prec l2 - l1
                                precs_2 = calc_precision(self.train_config.precs_to_calc, m2_tr, m1_tr,
                                                         self.train_data_model.dictionaries[(l2, l1)],
                                                         self.logger)
                            elif self.train_config.prec_calc_strat == 1:
                                m1 = copy.deepcopy(self.embeddings[l1])
                                m2 = copy.deepcopy(self.embeddings[l2])
                                m1.syn0 = np.dot(m1.syn0, T1)
                                m2.syn0 = np.dot(m2.syn0, T2)
                                # Prec l1 - l2
                                precs_1 = calc_precision(self.train_config.precs_to_calc, m1_tr, m2,
                                                         self.train_data_model.dictionaries[(l1, l2)],
                                                         self.logger)
                                # Prec l2 - l1
                                precs_2 = calc_precision(self.train_config.precs_to_calc, m2_tr, m1,
                                                         self.train_data_model.dictionaries[(l2, l1)],
                                                         self.logger)
                        if self.train_config.prec_calc_strat == 2 or self.train_config.prec_calc_strat == 3:
                            m1_tr = copy.deepcopy(self.valid_data_model.filtered_models[(l1, l2)])
                            m2_tr = copy.deepcopy(self.valid_data_model.filtered_models[(l2, l1)])
                            m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
                            m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
                            if self.train_config.prec_calc_strat == 2:
                                # Prec l1 - l2
                                precs_1 = calc_precision(self.train_config.precs_to_calc, m1_tr, m2_tr,
                                                         self.valid_data_model.dictionaries[(l1, l2)],
                                                         self.logger)
                                # Prec l2 - l1
                                precs_2 = calc_precision(self.train_config.precs_to_calc, m2_tr, m1_tr,
                                                         self.valid_data_model.dictionaries[(l2, l1)],
                                                         self.logger)
                            elif self.train_config.prec_calc_strat == 3:
                                m1 = copy.deepcopy(self.embeddings[l1])
                                m2 = copy.deepcopy(self.embeddings[l2])
                                m1.syn0 = np.dot(m1.syn0, T1)
                                m2.syn0 = np.dot(m2.syn0, T2)
                                # Prec l1 - l2
                                precs_1 = calc_precision(self.train_config.precs_to_calc, m1_tr, m2,
                                                         self.valid_data_model.dictionaries[(l1, l2)],
                                                         self.logger)
                                # Prec l2 - l1
                                precs_2 = calc_precision(self.train_config.precs_to_calc, m2_tr, m1,
                                                         self.valid_data_model.dictionaries[(l2, l1)],
                                                         self.logger)
                        e_prec_l.append(((l1, l2), precs_1))
                        e_prec_l.append(((l2, l1), precs_2))
                    self.logger.info(e_prec_l)
                    precs_arr.append(e_prec_l)
                fn = os.path.join(self.output_dir, 'T_{0}.pickle'.format(i))
                save_pickle(data=T, filename=fn)
        return T, lc_arr, precs_arr

    def run(self):
        T, lc_arr, precs_arr = self.train()
        loss_fn = os.path.join(self.output_dir, strings.LOSS_LOG_FN)
        prec_fn = os.path.join(self.output_dir, strings.PREC_LOG_FN)
        list_to_csv(lc_arr, loss_fn)
        save_pickle(precs_arr, prec_fn)
