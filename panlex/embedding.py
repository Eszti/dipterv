import datetime
import logging
import sys

from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText


class Embedding():
    def __init__(self):
        self.E = {}

    def get_dimension(self):
        raise NotImplementedError

    def get_sim(self, w1, w2):
        raise NotImplementedError

class Word2VecEmbedding(Embedding):
    @staticmethod
    def load_model(model_fn, model_type):
        logging.info('Loading model: {0}'.format(model_fn))
        if model_type == 'word2vec':
            model = KeyedVectors.load_word2vec_format(model_fn, binary=True)
        elif model_type == 'word2vec_txt':
            model = KeyedVectors.load_word2vec_format(model_fn, binary=False)
        else:
            raise Exception('Unknown model format')
        logging.info('Model loaded: {0}'.format(model_fn))
        return model

    def __init__(self, fn, model_type='word2vec'):
        self.fn = fn
        self.model_type = model_type
        self.model = Word2VecEmbedding.load_model(self.fn, self.model_type)

    def get_vec(self, w):
        if w in self.model:
            return self.model[w]
        return None

    def get_sim(self, w1, w2):
        if w1 in self.model and w2 in self.model:
            return self.model.similarity(w1, w2)
        else:
            return None

class FasttextEmbedding(Embedding):
    def __init__(self, fn):
        self.fn = fn
        self.model = FastText.load_fasttext_format(fn)

    def get_vec(self, w):
        if w in self.model.words:
            return self.model(w)
        return None

    def get_sim(self, w1, w2):
        if w1 in self.model and w2 in self.model:
            return self.model.similarity(w1, w2)
        else:
            return None

type_to_class = {
    'word2vec': Word2VecEmbedding,
    'word2vec_txt' : lambda fn: Word2VecEmbedding(fn, model_type='word2vec_txt'),
    'fasttext' : FasttextEmbedding
}

test_set = [
    ('king', 'queen'), ('cat', 'dog'), ('cup', 'coffee'), ('coffee', 'tea'),
    ('nice', 'kind'), ('dog', 'animal'), ('beautiful', 'ugly'), ('fawn', 'young')
]

def test():
    fn, e_type = sys.argv[1:3]
    e_class = type_to_class[e_type]
    model = e_class(fn)
    ts = datetime.datetime.now()
    print(ts)
    for w1, w2 in test_set:
        print("{0}\t{1}\t{2}".format(w1, w2, model.get_sim(w1, w2)))

if __name__ == "__main__":
    ts = datetime.datetime.now()
    print(ts)
    test()
    ts = datetime.datetime.now()
    print(ts)