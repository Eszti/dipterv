import sys

import os

sys.path.insert(0, 'utils')
sys.path.insert(0, 'base')

from io_helper import save_pickle, list_to_csv

import strings
from loggable import Loggable
import tensorflow as tf
import numpy as np


class TrainModel(Loggable):
    def __init__(self, train_config,
                 data_model_wrapper,
                 language_config, output_dir,
                 validation_model, plot_model,
                 cont_model):
        Loggable.__init__(self)
        self.train_config = train_config
        self.langs = language_config.langs
        self.logger.info('Language order: {0}'.format([(i, l) for i, l in enumerate(self.langs)]))

        # Train setup
        self.do_train = strings.TRAIN in data_model_wrapper.data_models.keys()
        if self.do_train:
            self.output_dir = os.path.join(output_dir, strings.TRAIN_OUTPUT_FOLDER_NAME)
            self.dim = data_model_wrapper.dim
            # Word pairs
            self.train_data_model = data_model_wrapper.data_models[strings.TRAIN]
            # Valid setup
            self.do_valid = validation_model is not None
            if self.do_valid:
                # Validation model
                self.validation_model = validation_model
                self.validation_model.set_datamodel(data_model_wrapper.data_models[strings.VALID])
            else:
                self.logger.info('Validation will be skipped !!! - no validation process is required')
            self.plot_model = plot_model
            self.cont_model = cont_model

    def _log_loss_after_epoch(self, sims, avg_sims, epoch_sims, i, loss_type):
        epoch_avg_ls = []
        for ((lang_pair), ls) in epoch_sims.items():
            lang_avg_arr = np.asarray(ls)
            lang_avg = np.average(lang_avg_arr)
            sims[lang_pair].append((i, lang_avg))
            epoch_avg_ls.append(lang_avg)
            self.logger.info('epoch:\t{0}\t{1}\tavg sims: {2}\t'.format(i, lang_pair, lang_avg))

            lang_sims_fn = os.path.join(self.output_dir, '{0}_{1}_{2}'
                                        .format(strings.SIM_LOG_FN, lang_pair[0], lang_pair[1]))
            list_to_csv(data=sims[lang_pair], filename=lang_sims_fn)

        epoch_avg_arr = np.asarray(epoch_avg_ls)
        epoch_avg = np.average(epoch_avg_arr)
        avg_sims.append((i, epoch_avg))
        self.logger.info('epoch:\t{0}\tavg sims: {1}\t- {2}'.format(i, epoch_avg, loss_type))

        avg_sims_fn = os.path.join(self.output_dir, strings.SIM_LOG_FN)
        list_to_csv(data=avg_sims, filename=avg_sims_fn)

    def train(self):
        nb_langs = len(self.langs)
        batch_size = self.train_config.batch_size

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
             # Load pretrained model
            if self.cont_model.cont:
                tf_T = tf.Variable(self.cont_model.T_loaded)

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
            sims = dict()
            avg_sims = []
            valid_done = False
            T_saved = False
            nb_train = len(self.train_data_model.word_pairs_dict[self.langs[0], self.langs[1]])
            print(nb_train)

            # Init
            done = []
            for l1 in self.langs:
                for l2 in self.langs:
                    lang_pair = tuple(sorted([l1, l2]))
                    if l1 == l2 or lang_pair in done:
                        continue
                    else:
                        sims[lang_pair] = []

            for epoch in range(self.train_config.epochs):
                T_saved = False
                svd_in_epoch = False
                epoch_sims = dict()

                for j_batch in range(int(round(1.0 * nb_train / batch_size))):
                    done = []
                    for l1 in self.langs:
                        for l2 in self.langs:
                            lang_pair = tuple(sorted([l1, l2]))
                            if l1 == l2 or lang_pair in done:
                                continue
                            if j_batch == 0:
                                epoch_sims[lang_pair] = []

                            idx_l1 = self.langs.index(l1)
                            idx_l2 = self.langs.index(l2)

                            wp_l = self.train_data_model.word_pairs_dict[l1, l2]
                            chosen_wps = wp_l[j_batch*batch_size : (j_batch+1)*batch_size]

                            W1, W2 = self.train_data_model.get_embeddings_for_batch( wp_l=chosen_wps,
                                                                                     dim=self.dim,
                                                                                     l1=l1, l2=l2)
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
                            epoch_sims[lang_pair].append(-l)
                            self.logger.debug('batch: {0} - loss: {1}'.format(j_batch, -l))
                            # print('batch: {0} - loss: {1} - lang: {2}'.format(j_batch, -l, lang_pair))
                            done.append(lang_pair)

                # Monitoring for learning curve
                self._log_loss_after_epoch(sims=sims,
                                           avg_sims=avg_sims,
                                           epoch_sims=epoch_sims,
                                           i=epoch,
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


