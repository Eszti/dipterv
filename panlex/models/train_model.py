import sys

import copy
import os

from debug_helper import get_smalls

sys.path.insert(0, 'utils')

from io_helper import save_pickle, list_to_csv
from math_helper import calc_precision, calc_precision_calc

import strings
from base.loggable import Loggable
import tensorflow as tf
import numpy as np


class TrainMModel(Loggable):
    def __init__(self, train_config, data_model_wrapper, language_config, output_dir, cont_model):
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
        # Set continue params
        self.cont_model = cont_model

    def _prec_eval_univ(self, prec_mode, l1, l2, T1, T2):
        precs_1 = []
        precs_2 = []
        self.logger.info('Univ space...')
        if prec_mode == 0 or prec_mode == 1:
            m1_tr = copy.deepcopy(self.train_data_model.filtered_models[(l1, l2)])
            m2_tr = copy.deepcopy(self.train_data_model.filtered_models[(l2, l1)])
            m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
            m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
            if prec_mode == 0:
                # Prec l1 - l2
                precs_1 = calc_precision(self.train_config.precs_to_calc, m1_tr, m2_tr,
                                         self.train_data_model.dictionaries[(l1, l2)],
                                         self.logger)
                # Prec l2 - l1
                precs_2 = calc_precision(self.train_config.precs_to_calc, m2_tr, m1_tr,
                                         self.train_data_model.dictionaries[(l2, l1)],
                                         self.logger)
            elif prec_mode == 1:
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
        if prec_mode == 2 or prec_mode == 3:
            m1_tr = copy.deepcopy(self.valid_data_model.filtered_models[(l1, l2)])
            m2_tr = copy.deepcopy(self.valid_data_model.filtered_models[(l2, l1)])
            m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
            m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
            if prec_mode == 2:
                # Prec l1 - l2
                precs_1 = calc_precision(self.train_config.precs_to_calc, m1_tr, m2_tr,
                                         self.valid_data_model.dictionaries[(l1, l2)],
                                         self.logger)
                # Prec l2 - l1
                precs_2 = calc_precision(self.train_config.precs_to_calc, m2_tr, m1_tr,
                                         self.valid_data_model.dictionaries[(l2, l1)],
                                         self.logger)
            elif prec_mode == 3:
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
        return precs_1, precs_2

    def _prec_eval_tar(self, prec_mode, l1, l2, T1, T1_inv, T2, T2_inv):
        self.logger.info('Target space...')
        # To init
        # Translation models
        W_trans_l1 = None
        W_trans_l2 = None
        i2w_trans_l1 = None
        i2w_trans_l2 = None
        # Validation models
        W_lookup_l1 = None
        W_lookup_l2 = None
        i2w_lookup_l1 = None
        i2w_lookup_l2 = None
        # Dictionaries
        dict_l1_to_l2 = None
        dict_l2_to_l1 = None
        # Translated models & dictionaries
        if prec_mode == 0 or prec_mode == 1:  # Test on train
            # Translated models
            m_l1 = self.train_data_model.filtered_models[(l1, l2)]
            m_l2 = self.train_data_model.filtered_models[(l2, l1)]
            W_univ_l1 = np.dot(m_l1.syn0, T1)
            W_univ_l1_n = W_univ_l1 / np.sqrt((W_univ_l1 ** 2).sum(1))[:, None]
            W_univ_l2 = np.dot(m_l2.syn0, T2)
            W_univ_l2_n = W_univ_l2 / np.sqrt((W_univ_l2 ** 2).sum(1))[:, None]
            W_trans_l1 = np.dot(W_univ_l1_n, T1_inv)
            W_trans_l2 = np.dot(W_univ_l2_n, T2_inv)
            i2w_trans_l1 = m_l1.index2word
            i2w_trans_l2 = m_l2.index2word
            # Dictionaries
            dict_l1_to_l2 = self.train_data_model.dictionaries[(l1, l2)]
            dict_l2_to_l1 = self.train_data_model.dictionaries[(l2, l1)]
        elif prec_mode == 2 or prec_mode == 3:  # Test on valid
            # Translated models
            m_l1 = self.valid_data_model.filtered_models[(l1, l2)]
            m_l2 = self.valid_data_model.filtered_models[(l2, l1)]
            # Todo: normalize!!!
            W_trans_l1 = np.dot(np.dot(m_l1.syn0, T1), T2_inv)
            W_trans_l2 = np.dot(np.dot(m_l2.syn0, T2), T1_inv)
            i2w_trans_l1 = m_l1.index2word
            i2w_trans_l2 = m_l2.index2word
            # Dictionaries
            dict_l1_to_l2 = self.valid_data_model.dictionaries[(l1, l2)]
            dict_l2_to_l1 = self.valid_data_model.dictionaries[(l2, l1)]

        # Validation models
        if prec_mode == 0 or prec_mode == 2:  # Lookup among only test words
            W_lookup_l1 = W_trans_l1
            W_lookup_l2 = W_trans_l2
            i2w_lookup_l1 = i2w_trans_l1
            i2w_lookup_l2 = i2w_trans_l2
        elif prec_mode == 1 or prec_mode == 3:  # Lookup among all embedding words
            m_val_l1 = self.embeddings[l1]
            m_val_l2 = self.embeddings[l2]
            W_lookup_l1 = m_val_l1.syn0
            W_lookup_l2 = m_val_l2.syn0
            i2w_lookup_l1 = m_val_l1.index2word
            i2w_lookup_l2 = m_val_l2.index2word

        precs_1 = calc_precision_calc(W_src=W_trans_l1, i2w_src=i2w_trans_l1,
                                      W_tar=W_lookup_l2, i2w_tar=i2w_lookup_l2,
                                      precs=self.train_config.precs_to_calc,
                                      dict_scr_2_tar=dict_l1_to_l2,
                                      logger=self.logger)
        precs_2 = calc_precision_calc(W_src=W_trans_l2, i2w_src=i2w_trans_l2,
                                      W_tar=W_lookup_l1, i2w_tar=i2w_lookup_l1,
                                      precs=self.train_config.precs_to_calc,
                                      dict_scr_2_tar=dict_l2_to_l1,
                                      logger=self.logger)
        return precs_1, precs_2

    def valid(self, l1, l2, T1, T1_inv, T2, T2_inv, eval_space):
        precs_1 = 0.0
        precs_2 = 0.0
        prec_mode = self.train_config.prec_calc_strat
        if strings.EVAL_SPACE_UNIV == eval_space:
            precs_1, precs_2 = self._prec_eval_univ(prec_mode=prec_mode,
                                                    l1=l1, l2=l2,
                                                    T1=T1, T2=T2)
        elif strings.EVAL_SPACE_TARGET == eval_space:
            precs_1, precs_2 = self._prec_eval_tar(prec_mode=prec_mode,
                                                   l1=l1, l2=l2,
                                                   T1=T1, T1_inv=T1_inv,
                                                   T2=T2, T2_inv=T2_inv)
        return precs_1, precs_2

    def _log_loss_after_epoch(self, loss_arr, lc_arr, i, loss_type):
        loss_np_arr = np.asarray(loss_arr)
        loss_epoch_avg = np.average(loss_np_arr)
        self.logger.info('epoch:\t{0}\tavg sims: {1}\t- {2}'.format(i, loss_epoch_avg, loss_type))
        lc_arr.append([i, loss_epoch_avg])

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
            if self.cont_model.cont:  # Load pretrained model
                tf_T = tf.Variable(self.cont_model.T_loaded)
            else:
                tf_T = tf.Variable(tf.truncated_normal([nb_langs, self.dim, self.dim]))

            # SVD reguralization
            tf_s1, tf_U1, tf_V1 = tf.svd(tf_T[tf_idx_l1], full_matrices=True, compute_uv=True)
            updated_1 = tf.assign(tf_T[tf_idx_l1], tf.matmul(tf_U1, tf_V1))
            tf_s2, tf_U2, tf_V2 = tf.svd(tf_T[tf_idx_l2], full_matrices=True, compute_uv=True)
            updated_2 = tf.assign(tf_T[tf_idx_l2], tf.matmul(tf_U2, tf_V2))

            # Loss
            tf_w1_u = tf.matmul(tf_w1, tf_T[tf_idx_l1])
            tf_w2_u = tf.matmul(tf_w2, tf_T[tf_idx_l2])
            tf_w1_u_n = tf.nn.l2_normalize(tf_w1_u, dim=1)
            tf_w2_u_n = tf.nn.l2_normalize(tf_w2_u, dim=1)
            loss = tf.matmul(tf_w1_u_n, tf.transpose(tf_w2_u_n))
            loss = -loss

            # Applying optimizer, Todo: try different optimizers!!
            # https://www.tensorflow.org/api_guides/python/train#Optimizers
            optimizer = tf.train.AdagradOptimizer(self.train_config.lr_base).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            j_iters = 0
            lc_u_arr = []
            lc_l1_arr = []
            lc_l2_arr = []
            precs_dict = dict()
            for i in range(self.train_config.epochs):
                loss_u_arr = []
                loss_l2_arr = []
                loss_l1_arr = []
                j_lang = 0
                for ((l1, l2), wp_l) in self.train_data_model.word_pairs_dict.items():
                    loss_U_arr_l = []
                    idx_l1 = self.langs.index(l1)
                    idx_l2 = self.langs.index(l2)
                    j_wp = 0
                    for (w1, w2) in wp_l:
                        emb1 = self.train_embeddings[l1][w1].reshape((1, 300))
                        emb2 = self.train_embeddings[l2][w2].reshape((1, 300))
                        calculated = False
                        if j_wp == 0:
                            if (self.train_config.svd_mode == 1 and i % self.train_config.svd_f == 0) or \
                                    (self.train_config.svd_mode == 2 and j_iters == 0):
                                _, l, _, _, T = session.run([optimizer, loss, updated_1, updated_2, tf_T],
                                                            feed_dict={tf_w1: emb1,
                                                                       tf_w2: emb2,
                                                                       tf_idx_l1: idx_l1,
                                                                       tf_idx_l2: idx_l2})
                                calculated = True

                        if not calculated:
                            _, l, T = session.run([optimizer, loss, tf_T],
                                                  feed_dict={tf_w1: emb1,
                                                             tf_w2: emb2,
                                                             tf_idx_l1: idx_l1,
                                                             tf_idx_l2: idx_l2})

                        # Calculate target loss in target space
                        if self.train_config.target_loss:
                            # T
                            T1 = T[idx_l1]
                            T2 = T[idx_l2]
                            # T inverses
                            T1_inv = np.linalg.inv(T1)
                            T2_inv = np.linalg.inv(T2)

                            # Loss L2 space
                            w1_u = np.dot(emb1, T1)
                            w1_u_n = w1_u / np.linalg.norm(w1_u)
                            w1_l2 = np.dot(w1_u_n, T2_inv)
                            w1_l2_n = w1_l2 / np.linalg.norm(w1_l2)
                            loss_l2 = np.dot(w1_l2_n, np.transpose(emb2))

                            # Loss L1 space
                            w2_u = np.dot(emb2, T2)
                            w2_u_n = w2_u / np.linalg.norm(w2_u)
                            w2_l1 = np.dot(w2_u_n, T1_inv)
                            w2_l1_n = w2_l1 / np.linalg.norm(w2_l1)
                            loss_l1 = tf.matmul(w2_l1_n, np.transpose(emb1))

                            # Loss in target spaces
                            loss_l1_arr.append(loss_l1[0][0])
                            loss_l2_arr.append(loss_l2[0][0])

                        j_iters += 1
                        j_wp += 1
                        loss_u_arr.append(-l[0][0])
                        loss_U_arr_l.append(-l[0][0])

                        if self.train_config.iters is not None and j_iters == self.train_config.iters:
                            break
                    j_lang += 1
                    if self.train_config.iters is not None and j_iters == self.train_config.iters:
                        break

                # Monitoring for learning curve
                self._log_loss_after_epoch(loss_arr=loss_u_arr, lc_arr=lc_u_arr, i=i, loss_type='universal space')
                if self.train_config.target_loss:
                    self._log_loss_after_epoch(loss_arr=loss_l1_arr, lc_arr=lc_l1_arr, i=i, loss_type='lang1 space')
                    self._log_loss_after_epoch(loss_arr=loss_l2_arr, lc_arr=lc_l2_arr, i=i, loss_type='lang2 space')

                for i, _ in enumerate(self.langs):
                    limit = 0.1       # Todo: config paraméterbe
                    T1 = T[i]
                    T2 = T[i]
                    get_smalls(T=T1, limit=limit, nb=i, logger=self.logger)
                    get_smalls(T=T2, limit=limit, nb=i, logger=self.logger)


                # Calculate precision
                if self.train_config.do_prec_calc:
                    for eval_space in self.train_config.prec_eval_spaces:
                        e_prec_l = []
                        for ((l1, l2), _) in self.train_data_model.word_pairs_dict.items():
                            self.logger.info('Calculating precision for {0}-{1}'.format(l1, l2))
                            # Get translations matrices
                            idx_l1 = self.langs.index(l1)
                            idx_l2 = self.langs.index(l2)
                            T1 = T[idx_l1]
                            T2 = T[idx_l2]
                            T1_inv = None
                            T2_inv = None
                            if strings.EVAL_SPACE_TARGET in self.train_config.prec_eval_spaces:
                                T1_inv = np.linalg.inv(T1)
                                T2_inv = np.linalg.inv(T2)

                            precs_1, precs_2 = self.valid(l1=l1, l2=l2,
                                                          T1=T1, T1_inv=T1_inv,
                                                          T2=T2, T2_inv=T2_inv,
                                                          eval_space=eval_space)

                            e_prec_l.append(((l1, l2), precs_1))
                            e_prec_l.append(((l2, l1), precs_2))
                        self.logger.info(e_prec_l)
                        if eval_space not in precs_dict.keys():
                            precs_dict[eval_space] = []
                        precs_dict[eval_space].append(e_prec_l)
                fn = os.path.join(self.output_dir, 'T_{0}.pickle'.format(i))
                save_pickle(data=T, filename=fn)
        return T, (lc_u_arr, lc_l1_arr, lc_l2_arr), precs_dict

    def run(self):
        T, lc_arr, precs_dict = self.train()
        loss_u_fn = os.path.join(self.output_dir, strings.LOSS_U_LOG_FN)
        list_to_csv(lc_arr[0], loss_u_fn)
        if self.train_config.target_loss:
            loss_l1_fn = os.path.join(self.output_dir, strings.LOSS_L1_LOG_FN)
            loss_l2_fn = os.path.join(self.output_dir, strings.LOSS_L2_LOG_FN)
            list_to_csv(lc_arr[1], loss_l1_fn)
            list_to_csv(lc_arr[2], loss_l2_fn)
        for k in precs_dict:
            prec_fn = os.path.join(self.output_dir, strings.PREC_LOG_FN, '_{}'.format(k))
            save_pickle(precs_dict[k], prec_fn)
