import sys
import json
import numpy as np
from gensim.models import KeyedVectors

from io_helper import load_pickle

sys.path.insert(0, 'base')
sys.path.insert(0, 'utils')

import strings
from loggable import Loggable

class EmbeddingModel(Loggable):
    def __init__(self):
        Loggable.__init__(self)
        self.syn0 = None
        self.index2word = None

    def normalize(self):
        self.syn0 /= np.sqrt((self.syn0 ** 2).sum(1))[:, None]
        self.logger.info('normalized, # > 0.001: {}'.format(
            len(np.where(abs(np.linalg.norm(self.syn0, axis=1) - 1) > 0.0001)[0]) ))

    def _read(self, fn, limit=None, lexicon=None, encoding='utf-8'):
        raise NotImplementedError

    def read(self, fn, limit=None, lexicon=None, encoding='utf-8'):
        self.logger.info('Reading embedding from {}'.format(fn))
        self._read(fn, limit, lexicon, encoding)
        self.normalize()
        self.logger.info('Syn0 size: {0}'.format(self.syn0.shape))

    def get(self, word):
        if word not in self.index2word:
            raise ValueError('Out of dictionary word: {}'.format(word))
        else:
            idx = self.index2word.index(word)
            return self.syn0[idx]

class PickleEmbedding(EmbeddingModel):
    def __init__(self):
        EmbeddingModel.__init__(self)

    def _read(self, fn, limit=None, lexicon=None, encoding='utf-8'):
        data = load_pickle(fn)
        self.syn0 = data[0]
        self.index2word = data[1]

class KeyedVectorEmbedding(EmbeddingModel):
    def __init__(self):
        EmbeddingModel.__init__(self)

    def get(self, word):
        print('keyedvec')
        return self.model[word]

    def _read(self, fn, limit=None, lexicon=None, encoding='utf-8'):
        model = KeyedVectors.load_word2vec_format(fn, binary=False, limit=limit, encoding=encoding)
        self.model = model
        self.syn0 = model.syn0
        self.index2word = model.index2word

class TextEmbedding(EmbeddingModel):
    def __init__(self):
        EmbeddingModel.__init__(self)

    def _read(self, fn, limit=None, lexicon=None, encoding='utf-8'):
        id2row = []
        def filter_lines(f):
            for i,line in enumerate(f):
                if limit is not None and i > limit:
                    break
                word = line.split()[0]
                if i != 0 and (lexicon is None or word in lexicon):
                    id2row.append(word)
                    yield line

        #get the number of columns
        with open(fn, encoding=encoding) as f:
            f.readline()
            ncols = len(f.readline().split())

        with open(fn, encoding=encoding) as f:
            m = np.matrix(np.loadtxt(filter_lines(f),
                          comments=None, usecols=range(1,ncols)))
        self.syn0 = np.asarray(m)
        self.index2word = id2row

string_to_type = {
    strings.PICKLE_EMB : PickleEmbedding,
    strings.TEXT_EMB : TextEmbedding,
    strings.KEYEDVEC_EMB : KeyedVectorEmbedding
}

class EmbeddingModelWrapper(Loggable):
    def __init__(self, language_config, embedding_config):
        Loggable.__init__(self)
        self.language_config = language_config
        self.embedding_config = embedding_config
        self._get_sil2fb_map()
        self._read_embeddings()

    def get_dim(self):
        for (_, e) in self.embeddings.items():
            return e.syn0.shape[1]

    def _get_sil2fb_map(self):
        with open(self.embedding_config.sil2fb_path) as f:
            self.sil2fb = json.load(f)

    def _read_embeddings(self):
        self.embeddings = dict()
        for sil in self.language_config.langs:
            code = sil
            if self.embedding_config.langcode_type == strings.FB:
                code = self.sil2fb[sil]
            fn = self.embedding_config.path.replace('*', code)

            emb_mod = string_to_type[self.embedding_config.format]()
            emb_mod.read(fn=fn, limit=self.embedding_config.limit, encoding=self.embedding_config.encoding)
            self.embeddings[sil] = emb_mod