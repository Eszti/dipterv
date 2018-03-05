import sys

import os

sys.path.insert(0, 'utils')

from io_helper import save_pickle, list_to_csv
from math_helper import get_embeddings_for_batch

import strings
from base.loggable import Loggable
import tensorflow as tf
import numpy as np


class TrainModel(Loggable):
    def __init__(self, train_config, data_model_wrapper, language_config, output_dir,
                 cont_model, validation_model, plot_model):
        Loggable.__init__(self)
        self.train_config = train_config
        self.batch_size = self.train_config.batch_size
        self.langs = language_config.langs
        self.logger.info('Language order: {0}'.format([(i, l) for i, l in enumerate(self.langs)]))
        self.dim = data_model_wrapper.dim
        # Train setup
        self.do_train = strings.TRAIN in data_model_wrapper.data_models.keys()
        if self.do_train:
            self.train_data_model = data_model_wrapper.data_models[strings.TRAIN]
            # Getting embeddings
            self.train_embeddings = self.train_data_model.filtered_input_embeddings
            self.embeddings = data_model_wrapper.embedding_model.embeddings
            self.output_dir = os.path.join(output_dir, strings.TRAIN_OUTPUT_FOLDER_NAME)
            # Set continue params
            self.cont_model = cont_model
            # Valid setup
            self.do_valid = validation_model is not None
            if self.do_valid:
                # Validation model
                self.validation_model = validation_model
            else:
                self.logger.info('Validation will be skipped !!! - no validation process is required')
            self.plot_model = plot_model

    def _log_loss_after_epoch(self, loss_arr, lc_arr, i, loss_type):
        loss_np_arr = np.asarray(loss_arr)
        loss_epoch_avg = np.average(loss_np_arr)
        self.logger.info('epoch:\t{0}\tavg sims: {1}\t- {2}'.format(i, loss_epoch_avg, loss_type))
        lc_arr.append([i, loss_epoch_avg])
        loss_u_fn = os.path.join(self.output_dir, strings.SIM_LOG_FN)
        list_to_csv(data=lc_arr, filename=loss_u_fn)

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
            loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf_w1_u_n, tf_w2_u_n), axis=1))
            loss = -loss

            # Applying optimizer, Todo: try different optimizers!!
            # https://www.tensorflow.org/api_guides/python/train#Optimizers
            optimizer = tf.train.AdagradOptimizer(self.train_config.lr_base).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            sim_u_arr = []
            valid_done = False
            T_saved = False
            for epoch in range(self.train_config.epochs):
                T_saved = False
                svd_in_epoch = False
                loss_u_arr = []

                for (j_lang, ((l1, l2), wp_l)) in enumerate(self.train_data_model.word_pairs_dict.items()):
                    loss_U_arr_l = []
                    idx_l1 = self.langs.index(l1)
                    idx_l2 = self.langs.index(l2)
                    for j_batch in range(int(round(1.0 * len(wp_l) / self.batch_size))):
                        chosen_wps = wp_l[j_batch*self.batch_size : (j_batch+1)*self.batch_size]

                        W1, W2 = get_embeddings_for_batch(emb_dict=self.train_embeddings, wp_l=chosen_wps,
                                                          dim=self.dim, l1=l1, l2=l2)
                        svd_done = False
                        if j_batch == 0:
                            if (self.train_config.svd_mode == 1 and epoch % self.train_config.svd_f == 0) or \
                                    (self.train_config.svd_mode == 2 and epoch == 0):
                                _, l, _, _, T = session.run([optimizer, loss, updated_1, updated_2, tf_T],
                                                            feed_dict={tf_w1: W1,
                                                                       tf_w2: W2,
                                                                       tf_idx_l1: idx_l1,
                                                                       tf_idx_l2: idx_l2})
                                svd_done = True
                                svd_in_epoch = True
                        if not svd_done:
                            _, l, T = session.run([optimizer, loss, tf_T],
                                                  feed_dict={tf_w1: W1,
                                                             tf_w2: W2,
                                                             tf_idx_l1: idx_l1,
                                                             tf_idx_l2: idx_l2})
                        loss_u_arr.append(-l)
                        loss_U_arr_l.append(-l)
                        self.logger.debug('batch: {0} - loss: {1}'.format(j_batch, -l))

                # Monitoring for learning curve
                self._log_loss_after_epoch(loss_arr=loss_u_arr,
                                           lc_arr=sim_u_arr, i=epoch,
                                           loss_type='universal space')
                if self.do_valid:
                    # Validate
                    valid_done = self.validation_model.do_validation(svd_done=svd_in_epoch, epoch=epoch, T=T)

                # Save T matrix
                if (not self.train_config.save_only_on_valid) or (self.train_config.save_only_on_valid and valid_done):
                    fn = os.path.join(self.output_dir, 'T_{0}.pickle'.format(epoch))
                    save_pickle(data=T, filename=fn)
                    T_saved = True

            if self.do_valid:
                # Valid after all
                if not valid_done:
                    self.validation_model.do_validation(svd_done=True, epoch=self.train_config.epochs, T=T)
            # Save after all
            if not T_saved:
                fn = os.path.join(self.output_dir, 'T_{0}.pickle'.format(self.train_config.epochs))
                save_pickle(data=T, filename=fn)

    def run(self):
        if self.do_train:
            self.train()
            self.plot_model.plot_progress()
        else:
            self.logger.info('Skipping training')


