import json
import pickle

import numpy as np
import os
import sys
from gensim.models import KeyedVectors

sys.path.insert(0, 'utils')
sys.path.insert(0, 'base')

import strings
from loggable import Loggable


class DataModel(Loggable):
    def __init__(self, language_config, data_model_config, embedding_model):
        Loggable.__init__(self)
        # Configs
        self.language_config = language_config
        self.data_model_config = data_model_config
        # Initialize
        # lang1 - Lang2 : word pair list
        self.word_pairs_dict = dict()
        self.get_word_pairs_dict()
        if data_model_config.emb_dir is not None:       # in case of we have saved training words embedding separately
            self.logger.info('embedding is read from : {}'.format(data_model_config.emb_dir))
            self._read_filtered_embedding(data_model_config.emb_dir)
            embeddings = self.training_embeddings
        else:
            self.logger.info('original embedding is used'.format(data_model_config.emb_dir))
            embeddings = embedding_model.embeddings
        # Lang1 - Lang2 : Lang1 - Lang2 dictionary
        self.dictionaries = dict()
        self.get_two_lang_dictionaries(embeddings)
        # Lang : reduced amoung of word embeddings
        self.filtered_models = dict()
        self.get_filtered_models(embeddings, embedding_model.get_dim())

    def get_word_pairs_dict(self):
        done = set()
        for lang1 in self.language_config.langs:
            for lang2 in self.language_config.langs:
                lang_pair = tuple(sorted([lang1, lang2]))
                if lang1 == lang2 or lang_pair in done:
                    continue
                done.add(lang_pair)
                l1 = lang_pair[0]
                l2 = lang_pair[1]
                fn = os.path.join(self.data_model_config.dir, '{0}_{1}.tsv'.format(l1, l2))
                self.logger.info('Reading word pair file: {0}'.format(fn))
                data = self._read_word_pairs_tsv(fn, self.data_model_config.idx1, self.data_model_config.idx2, False)
                self.word_pairs_dict[lang_pair] = data
                self.logger.info('Number of word pairs found: {0}'.format(len(data)))

    # Read word pairs from tsv
    def _read_word_pairs_tsv(self, fn, id1, id2, header=True):
        with open(fn) as f:
            lines = f.readlines()
            data = [(line.split()[id1], line.split()[id2]) for i, line in enumerate(lines) if i > 0 or header == False]
        return data

    def _read_filtered_embedding(self, emb_dir):
        self.training_embeddings = dict()
        for l in self.language_config.langs:
            fn = os.path.join(emb_dir, '{}.pickle'.format(l))
            with open(fn, 'rb') as f:
                data = pickle.load(f)
            self.training_embeddings[l] = data

    def _get_not_found_list(self, vocab, embedding):
        nf_list = []
        for i, w in enumerate(vocab):
            # Check if there's an embedding to the word
            if w not in embedding:
                nf_list.append(w)
        return nf_list

    def get_two_lang_dictionaries(self, embeddings):
        updated_word_pairs = dict()
        for ((l1, l2), wp_l) in self.word_pairs_dict.items():
            self.logger.info('Processing {0}-{1}...'.format(l1, l2))
            # Find words without embeddings
            [l1_vocab, l2_vocab] = zip( *wp_l)
            l1_vocab = list(set(l1_vocab))
            l2_vocab = list(set(l2_vocab))
            self.logger.info('Words in {0}: {1}'.format(l1, len(l1_vocab)))
            self.logger.info('Words in {0}: {1}'.format(l2, len(l2_vocab)))
            nf_l1 = self._get_not_found_list(vocab=l1_vocab, embedding=embeddings[l1])
            self.logger.info('Words not found in embedding {0}: {1}'.format(l1, len(nf_l1)))
            self.logger.debug(nf_l1)
            nf_l2 = self._get_not_found_list(vocab=l2_vocab, embedding=embeddings[l2])
            self.logger.info('Words not found in embedding {0}: {1}'.format(l2, len(nf_l2)))
            self.logger.debug(nf_l2)
            # Update word list
            self.logger.info('Updating word pair list {0}-{1}'.format(l1, l2))
            updated_wp_l = [(w1, w2) for (w1, w2) in wp_l if w1 not in nf_l1 and w2 not in nf_l2]
            self.logger.info('Word pairs list legth: {0} ->  {1} '.format(len(wp_l), len(updated_wp_l)))
            updated_word_pairs[(l1, l2)] = updated_wp_l
            # Create dictioary
            self.logger.info('Creating dictionary for: {0}-{1}'.format(l1, l2))
            l12, l21 = self._wp_list_2_dict(updated_wp_l)
            self.dictionaries[(l1, l2)] = l12
            self.dictionaries[(l2, l1)] = l21
            self.logger.info('# word in: {0}-{1}:\t{2}'.format(l1.upper(), l2, len(l12)))
            self.logger.info('# word in: {0}-{1}:\t{2}'.format(l2.upper(), l1, len(l21)))
        self.word_pairs_dict = updated_word_pairs

    # Get dictionary fromm wordlist
    def _wp_list_2_dict(self, wp_l):
        l12 = dict()
        l21 = dict()
        for (w1, w2) in wp_l:
            if w1 not in l12:
                l12[w1] = [w2]
            else:
                l12[w1].append(w2)
            if w2 not in l21:
                l21[w2] = [w1]
            else:
                l21[w2].append(w1)
        return l12, l21

    def get_filtered_models(self, embeddings, dim):
        for ((l1, l2), d) in self.dictionaries.items():
            filtered_mod = KeyedVectors()
            filtered_mod.index2word = list(d.keys())
            filtered_mod.syn0 = np.ndarray(shape=(len(filtered_mod.index2word), dim), dtype=np.float32)
            # Adding embedding to train model
            for i, w in enumerate(filtered_mod.index2word):
                filtered_mod.syn0[i, :] = embeddings[l1][w]
            self.logger.info('Filtered model: {0}-{1} contains {2} words'.
                             format(l1.upper(), l2, len(filtered_mod.syn0)))
            self.filtered_models[(l1, l2)] = filtered_mod

class EmbeddingModel(Loggable):
    def __init__(self, language_config, embedding_config):
        Loggable.__init__(self)
        self.language_config = language_config
        self.embedding_config = embedding_config
        self._get_sil2fb_map()
        self._read_embeddings()

    def get_dim(self):
        for (_, e) in self.embeddings.items():
            return e.vector_size

    def _get_sil2fb_map(self):
        with open(self.embedding_config.sil2fb_path) as f:
            self.sil2fb = json.load(f)

    def _read_embeddings(self):
        self.embeddings = dict()
        for sil in self.language_config.langs:
            fb = self.sil2fb[sil]
            fn = self.embedding_config.path.replace('*', fb)
            self.logger.info('Reading embedding from {}'.format(fn))
            model = KeyedVectors.load_word2vec_format(fn, binary=False, limit=self.embedding_config.limit)
            model.syn0 /= np.sqrt((model.syn0 ** 2).sum(1))[:, None]
            self.embeddings[sil] = model
            self.logger.info('# word:\t{}'.format(len(model.syn0)))

class DataModelWrapper(Loggable):
    def __init__(self, data_wrapper_config, embedding_config, language_config):
        Loggable.__init__(self)
        self.data_models = dict()
        self.embedding_model = EmbeddingModel(language_config=language_config, embedding_config=embedding_config)
        for (key, data_model_config) in data_wrapper_config.data_configs.items():
            self.logger.info('Creating data model for {} ...'.format(key.upper()))
            self.data_models[key] = DataModel(language_config=language_config,
                                              data_model_config=data_model_config,
                                              embedding_model=self.embedding_model)
        self.dim = self.embedding_model.get_dim()
        self.logger.info('Vector dimension: {}'.format(self.dim))
        self.training_embeddings = self.data_models[strings.TRAIN].training_embeddings
