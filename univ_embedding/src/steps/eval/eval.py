import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Eval():
    def __init__(self, name):
        self.name = name

    def _get_cos_sim_mx(self, emb):
        cnt = emb.shape[0]
        mx = np.ndarray(shape=(cnt, cnt), dtype=np.float32)
        for i in range(0, cnt):
            for j in range(0, i + 1):
                sim = cosine_similarity(emb[i].reshape(1, -1), emb[j].reshape(1, -1))
                mx[i][j] = sim
                mx[j][i] = sim
        return mx

    def _evalute(self, input):
        raise NotImplementedError

    def evalute(self, input):
        logging.info('Evaluation {} has started'.format(self.name))
        output = self._evalute(input)
        logging.info('Evaluation {} has finished'.format(self.name))
        return output

    def save_result(self, header, result, fn):
        raise NotImplementedError