import sys

import os

from io_helper import list_to_csv

sys.path.insert(0, 'utils')
sys.path.insert(0, 'base')

from debug_helper import get_smalls
from math_helper import calc_precision, gather, calc_loss, get_indexes_of_wplist

import strings
from loggable import Loggable
import numpy as np


class TestBaseModel(Loggable):
    def __init__(self, model_config, language_config, output_dir, type):
        Loggable.__init__(self)
        self.model_config = model_config
        self.langs = language_config.langs

        # Counters
        self.sim_lang_wise = dict()
        self.sim_cumm = []
        self.precs_lang_wise = dict()

        # Output folder
        folder_name = ''
        if type == strings.VALID:
            folder_name = strings.VALID_OUTPUT_FOLDER_NAME
        elif type == strings.TEST:
            folder_name = strings.TEST_OUTPUT_FOLDER_NAME
        self.output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(self.output_dir)

    def set_datamodel(self, data_model):
        self.data_model = data_model
        self.gathered_embeddings_indices = self._create_gathered_embeddings_idices()
        # Gold dictionary
        self.gold_dict = self.data_model.get_gold_dictionary()
        self.filtered_model_dict = self.data_model.get_filtered_models_dict()

    def _save_valid_sim_lang_wise(self):
        for ((l1, l2), ls) in self.sim_lang_wise.items():
            fn = os.path.join(self.output_dir, '{0}_{1}_{2}'.format(strings.SIM_LOG_FN, l1, l2))
            list_to_csv(data=ls, filename=fn)

    def _save_valid_sim(self):
        fn = os.path.join(self.output_dir, strings.SIM_LOG_FN)
        list_to_csv(data=self.sim_cumm, filename=fn)

    def _save_precs(self):
        for ((l1, l2), ls) in self.precs_lang_wise.items():
            fn = os.path.join(self.output_dir, '{0}_{1}_{2}'.format(strings.PREC_LOG_FN, l1, l2))
            list_to_csv(data=ls, filename=fn)

    def _create_gathered_embeddings_idices(self):
        gathered_embeddings_indices = dict()
        for ((l1, l2), wp_l) in self.data_model.word_pairs_dict.items():
            M1 = self.data_model.embeddings[l1]
            M2 = self.data_model.embeddings[l2]
            l1_idxs, l2_idxs = get_indexes_of_wplist(wp_l=wp_l, emb_l1=M1, emb_l2=M2)
            gathered_embeddings_indices[(l1, l2)] = [l1_idxs, l2_idxs]
        return gathered_embeddings_indices

    def do(self, epoch, T):
        epoch_valid_sim_cumm = []
        for ((l1, l2), [idxs1, idxs2]) in self.gathered_embeddings_indices.items():
            idx_l1 = self.langs.index(l1)
            idx_l2 = self.langs.index(l2)
            T1 = T[idx_l1]
            T2 = T[idx_l2]
            # Calculate valid loss
            if self.model_config.calc_loss:
                # One word can occure several times
                M1 = self.data_model.embeddings[l1].syn0
                M2 = self.data_model.embeddings[l2].syn0
                W1 = gather(M=M1, idxs=idxs1)
                W2 = gather(M=M2, idxs=idxs2)
                if (l1, l2) not in self.sim_lang_wise.keys():
                    self.sim_lang_wise[(l1, l2)] = []
                M1 = np.dot(W1, T1)
                M1 /= np.sqrt((M1 ** 2).sum(1))[:, None]
                M2 = np.dot(W2, T2)
                M2 /= np.sqrt((M2 ** 2).sum(1))[:, None]
                avg_cos = calc_loss(M1=M1, M2=M2)
                self.logger.info('Avg valid loss {0}-{1}: {2}'.format(l1, l2, avg_cos))
                # Lang-pair stat
                self.sim_lang_wise[(l1, l2)].append((epoch, avg_cos))
                # Epoch stat
                epoch_valid_sim_cumm.append(avg_cos)

            # Calculate precision
            if self.model_config.do_prec_calc:
                m1 = self.filtered_model_dict[(l1, l2)]
                m2 = self.filtered_model_dict[(l2, l1)]
                # One word occurs only one time
                W1 = m1.syn0
                W2 = m2.syn0

                W1_univ = np.dot(W1, T1)
                W2_univ = np.dot(W2, T2)
                lookup1_univ = np.dot(self.data_model.embeddings[l1].syn0, T1)
                lookup2_univ = np.dot(self.data_model.embeddings[l2].syn0, T2)

                dict12 = self.gold_dict[(l1, l2)]
                dict21 = self.gold_dict[(l2, l1)]

                # Prec l1 - l2
                precs_1 = calc_precision(W_src=W1_univ, i2w_src=m1.index2word,
                                         W_tar=lookup2_univ, i2w_tar=self.data_model.embeddings[l2].index2word,
                                         precs=self.model_config.precs_to_calc,
                                         dict_scr_2_tar=dict12, logger=self.logger)
                self.logger.info('Precs: {0}-{1}: {2}'.format(l1, l2, precs_1))
                # Prec l2 - l1
                precs_2 = calc_precision(W_src=W2_univ, i2w_src=m2.index2word,
                                         W_tar=lookup1_univ, i2w_tar=self.data_model.embeddings[l1].index2word,
                                         precs=self.model_config.precs_to_calc,
                                         dict_scr_2_tar=dict21, logger=self.logger)
                self.logger.info('Precs: {0}-{1}: {2}'.format(l2, l1, precs_2))

                if (l1, l2) not in self.precs_lang_wise.keys():
                    self.precs_lang_wise[(l1, l2)] = [[0] + self.model_config.precs_to_calc]
                if (l2, l1) not in self.precs_lang_wise.keys():
                    self.precs_lang_wise[(l2, l1)] = [[0] + self.model_config.precs_to_calc]
                self.precs_lang_wise[(l1, l2)].append([epoch] + precs_1)
                self.precs_lang_wise[(l2, l1)].append([epoch] + precs_2)

        # Calculate small singular values
        if self.model_config.calc_small_sing:
            for i, _ in enumerate(self.langs):
                T1 = T[i]
                get_smalls(T=T1, limit=self.model_config.limit, nb=i, logger=self.logger)

        # Calculate average similarity over all the languages
        epoch_valid_sim_cumm_avg = np.average(np.asanyarray(epoch_valid_sim_cumm))
        self.sim_cumm.append((epoch, epoch_valid_sim_cumm_avg))

        # Save files
        if self.model_config.calc_loss:
            self._save_valid_sim_lang_wise()
            self._save_valid_sim()
        if self.model_config.do_prec_calc:
            self._save_precs()