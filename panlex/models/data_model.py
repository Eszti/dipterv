import collections
import os
import sys
import numpy as np

sys.path.insert(0, 'utils')
sys.path.insert(0, 'base')

from loggable import Loggable
from embedding_models import EmbeddingModelWrapper, EmbeddingModel


class DataModel(Loggable):
    def __init__(self, language_config, data_model_config, embedding_model_wrapper):
        Loggable.__init__(self)
        # Configs
        self.language_config = language_config
        self.data_model_config = data_model_config
        if self.data_model_config.emb is not None:
            own_embedding_model_wrapper = EmbeddingModelWrapper(
                language_config=language_config, embedding_config=self.data_model_config.embedding_config)
            self.embeddings = own_embedding_model_wrapper.embeddings
        else:
            self.embeddings = embedding_model_wrapper.embeddings
        # { (l1, l2) : [(w1, w2)] }
        self.word_pairs_dict = self._get_word_pairs_dict(self.embeddings)

    def _get_word_pairs_dict(self, emb_dict):
        word_pairs_dict = dict()
        done = set()
        for lang1 in self.language_config.langs:
            for lang2 in self.language_config.langs:
                lang_pair = tuple(sorted([lang1, lang2]))
                if lang1 == lang2 or lang_pair in done:
                    continue
                done.add(lang_pair)
                l1 = lang_pair[0]
                l2 = lang_pair[1]
                # Read tsv files
                fn = os.path.join(self.data_model_config.dir, '{0}_{1}.tsv'.format(l1, l2))
                self.logger.info('Reading word pair file: {0}'.format(fn))
                header = False
                with open(fn) as f:
                    lines = f.readlines()
                    data = [(line.split()[ self.data_model_config.idx1], line.split()[ self.data_model_config.idx2])
                            for i, line in enumerate(lines) if i > 0 or header == False]
                self.logger.info('Number of word pairs: {0}'.format(len(data)))
                # Get valid data
                emb1 = emb_dict[l1]
                emb2 = emb_dict[l2]
                valid_data = [(w1, w2) for (w1, w2) in data if w1 in emb1.index2word and w2 in emb2.index2word]
                self.logger.info('Number of valid word pairs: {0}'.format(len(valid_data)))
                word_pairs_dict[(l1, l2)] = valid_data
        return word_pairs_dict

    def get_gold_dictionary(self):
        gold_dict = dict()
        done = set()
        for lang1 in self.language_config.langs:
            for lang2 in self.language_config.langs:
                lang_pair = tuple(sorted([lang1, lang2]))
                if lang1 == lang2 or lang_pair in done:
                    continue
                done.add(lang_pair)
                gold_l1 = collections.defaultdict(set)
                gold_l2 = collections.defaultdict(set)
                for (w1, w2) in self.word_pairs_dict[(lang1, lang2)]:
                    gold_l1[w1].add(w2)
                    gold_l2[w2].add(w1)
                # lang1 -> [lang2]
                gold_dict[(lang1, lang2)] = gold_l1
                gold_dict[(lang2, lang1)] = gold_l2
        return gold_dict

    def get_embeddings_for_batch(self, wp_l, dim, l1, l2):
        nb_rows = len(wp_l)
        emb1 = np.ndarray(shape=(nb_rows, dim))
        emb2 = np.ndarray(shape=(nb_rows, dim))
        for (i, (w1, w2)) in enumerate(wp_l):
            emb1[i, :] = self.embeddings[l1].get(w1)
            emb2[i, :] = self.embeddings[l2].get(w2)
        return emb1, emb2

    def _get_emb(self, words, lang):
        W = np.ndarray(shape=(len(words), self.embeddings[lang].syn0[0].shape[0]))
        for i, w in enumerate(words):
            W[i, :] = self.embeddings[lang].get(w)
        emb = EmbeddingModel()
        emb.index2word = words
        emb.syn0 = W
        return emb

    def get_filtered_models_dict(self):
        filtered_models_dict = dict()
        done = set()
        for lang1 in self.language_config.langs:
            for lang2 in self.language_config.langs:
                lang_pair = tuple(sorted([lang1, lang2]))
                if lang1 == lang2 or lang_pair in done:
                    continue
                done.add(lang_pair)
                words1, words2 = zip(*self.word_pairs_dict[(lang1, lang2)])
                words1 = list(set(words1))
                words2 = list(set(words2))
                emb_l1 = self._get_emb(words1, lang1)
                emb_l2 = self._get_emb(words2, lang2)
                filtered_models_dict[(lang1, lang2)] = emb_l1
                filtered_models_dict[(lang2, lang1)] = emb_l2
        return filtered_models_dict

class DataModelWrapper(Loggable):
    def __init__(self, data_wrapper_config, embedding_config, language_config):
        Loggable.__init__(self)
        self.data_models = dict()
        embedding_model_wrapper = EmbeddingModelWrapper(language_config=language_config,
                                                             embedding_config=embedding_config)
        for (key, data_model_config) in data_wrapper_config.data_configs.items():
            self.logger.info('Creating data model for {} ...'.format(key.upper()))
            self.data_models[key] = DataModel(language_config=language_config,
                                              data_model_config=data_model_config,
                                              embedding_model_wrapper=embedding_model_wrapper)
        self.dim = embedding_model_wrapper.get_dim()