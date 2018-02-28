import sys

import os

from io_helper import list_to_csv

sys.path.insert(0, 'utils')

from debug_helper import get_smalls
from math_helper import calc_precision, gather, calc_loss, get_indexes_of_wplist

import strings
from base.loggable import Loggable
import numpy as np


class ValidModel(Loggable):
    def __init__(self, valid_config, data_model_wrapper, language_config, output_dir):
        Loggable.__init__(self)
        self.valid_config = valid_config
        self.langs = language_config.langs
        self.valid_data_model = data_model_wrapper.data_models[strings.VALID]
        self.embeddings = data_model_wrapper.embedding_model.embeddings
        self.gathered_embeddings_indices = self._create_gathered_embeddings_idices()
        self.valid_sim_lang_wise = dict()
        self.valid_sim_cumm = []
        self.precs_lang_wise = dict()
        self.output_dir = os.path.join(output_dir, strings.VALID_OUTPUT_FOLDER_NAME)
        os.makedirs(self.output_dir)

    def _create_gathered_embeddings_idices(self):
        gathered_embeddings_indices = dict()
        for ((l1, l2), wp_l) in self.valid_data_model.word_pairs_dict.items():
            l1_idxs, l2_idxs = get_indexes_of_wplist(wp_l=wp_l, embeddings=self.embeddings, l1=l1, l2=l2)
            gathered_embeddings_indices[(l1, l2)] = [l1_idxs, l2_idxs]
        return gathered_embeddings_indices

    def _save_valid_sim_lang_wise(self):
        for ((l1, l2), ls) in self.valid_sim_lang_wise.items():
            fn = os.path.join(self.output_dir, '{0}_{1}_{2}'.format(strings.VALID_SIM_FN, l1, l2))
            list_to_csv(data=ls, filename=fn)

    def _save_valid_sim(self):
        fn = os.path.join(self.output_dir, strings.VALID_SIM_FN)
        list_to_csv(data=self.valid_sim_cumm, filename=fn)

    def _save_precs(self):
        for ((l1, l2), ls) in self.precs_lang_wise.items():
            fn = os.path.join(self.output_dir, '{0}_{1}_{2}'.format(strings.PREC_LOG_FN, l1, l2))
            list_to_csv(data=ls, filename=fn)

    def do_validation(self, svd_done, epoch, T):
        if svd_done or epoch % self.valid_config.do_valid_on == 0:
            epoch_valid_sim_cumm = []
            for ((l1, l2), [idxs1, idxs2]) in self.gathered_embeddings_indices.items():
                idx_l1 = self.langs.index(l1)
                idx_l2 = self.langs.index(l2)
                T1 = T[idx_l1]
                T2 = T[idx_l2]
                # Calculate valid loss
                if self.valid_config.calc_valid_loss:
                    # One word can occure several times
                    W1 = gather(M=self.embeddings[l1].syn0, idxs=idxs1)
                    W2 = gather(M=self.embeddings[l2].syn0, idxs=idxs2)
                    if (l1, l2) not in self.valid_sim_lang_wise.keys():
                        self.valid_sim_lang_wise[(l1, l2)] = []
                    M1 = np.dot(W1, T1)
                    M1 /= np.sqrt((M1 ** 2).sum(1))[:, None]
                    M2 = np.dot(W2, T2)
                    M2 /= np.sqrt((M2 ** 2).sum(1))[:, None]
                    avg_cos = calc_loss(M1=M1, M2=M2)
                    self.logger.info('Avg valid loss {0}-{1}: {2}'.format(l1, l2, avg_cos))
                    # Lang-pair stat
                    self.valid_sim_lang_wise[(l1, l2)].append((epoch, avg_cos))
                    # Epoch stat
                    epoch_valid_sim_cumm.append(avg_cos)

                # Calculate precision
                if self.valid_config.do_prec_calc:
                    m1 = self.valid_data_model.filtered_models[(l1, l2)]
                    m2 = self.valid_data_model.filtered_models[(l2, l1)]
                    # One word occurs only one time
                    W1 = self.valid_data_model.filtered_models[(l1, l2)].syn0
                    W2 = self.valid_data_model.filtered_models[(l2, l1)].syn0

                    W1_univ = np.dot(W1, T1)
                    W2_univ = np.dot(W2, T2)
                    lookup1_univ = np.dot(self.embeddings[l1].syn0, T1)
                    lookup2_univ = np.dot(self.embeddings[l2].syn0, T2)

                    dict12 = self.valid_data_model.dictionaries[(l1, l2)]
                    dict21 = self.valid_data_model.dictionaries[(l2, l1)]

                    # Prec l1 - l2
                    precs_1 = calc_precision(W_src=W1_univ, i2w_src=m1.index2word,
                                             W_tar=lookup2_univ, i2w_tar=self.embeddings[l2].index2word,
                                             precs=self.valid_config.precs_to_calc,
                                             dict_scr_2_tar=dict12, logger=self.logger)
                    self.logger.info('Precs: {0}-{1}: {2}'.format(l1, l2, precs_1))
                    # Prec l2 - l1
                    precs_2 = calc_precision(W_src=W2_univ, i2w_src=m2.index2word,
                                             W_tar=lookup1_univ, i2w_tar=self.embeddings[l1].index2word,
                                             precs=self.valid_config.precs_to_calc,
                                             dict_scr_2_tar=dict21, logger=self.logger)
                    self.logger.info('Precs: {0}-{1}: {2}'.format(l2, l1, precs_2))

                    if (l1, l2) not in self.precs_lang_wise.keys():
                        self.precs_lang_wise[(l1, l2)] = [[0] + self.valid_config.precs_to_calc]
                    if (l2, l1) not in self.precs_lang_wise.keys():
                        self.precs_lang_wise[(l2, l1)] = [[0] + self.valid_config.precs_to_calc]
                    self.precs_lang_wise[(l1, l2)].append([epoch] + precs_1)
                    self.precs_lang_wise[(l2, l1)].append([epoch] + precs_2)

                # Calculate small singular values
                if self.valid_config.calc_small_sing:
                    for i, _ in enumerate(self.langs):
                        T1 = T[i]
                        get_smalls(T=T1, limit=self.valid_config.limit, nb=i, logger=self.logger)

            # Calculate average similarity over all the languages
            epoch_valid_sim_cumm_avg = np.average(np.asanyarray(epoch_valid_sim_cumm))
            self.valid_sim_cumm.append((epoch, epoch_valid_sim_cumm_avg))

            # Save files
            if self.valid_config.calc_valid_loss:
                self._save_valid_sim_lang_wise()
                self._save_valid_sim()
            if self.valid_config.do_prec_calc:
                self._save_precs()