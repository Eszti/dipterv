import logging
import numpy as np
import scipy
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

class Evaluation():
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def init_for_eval(self):
        raise NotImplementedError

    def _log_cfg(self, section, key, value):
        logging.info('Conf param read: [{0}]: {1} - {2}'.format(section, key, value))

    def get(self, cfg_key, type=None, section=None):
        if section is None:
            section = self.name
        if type is None:
            value = self.config.get(section, cfg_key)
        elif type == 'int':
            value = self.config.getint(section, cfg_key)
        elif type == 'float':
            value = self.config.getfloat(section, cfg_key)
        elif type == 'boolean':
            value = self.config.getboolean(section, cfg_key)
        self._log_cfg(section, cfg_key, value)
        return value

    def _get_cos_sim_mx(self, emb):
        mx = 1 - spatial.distance.cdist(emb, emb, 'cosine')
        return mx

    def _corr_cos_sims(self, cos1, cos2):
        flat1 = cos1[np.triu_indices(cos1.shape[0])]
        flat2 = cos2[np.triu_indices(cos2.shape[0])]
        corr = scipy.stats.pearsonr(flat1, flat2)
        return corr

    def _evalute(self, input):
        raise NotImplementedError

    def evalute(self, input):
        logging.info('Evaluation {} has started'.format(self.name))
        self.init_for_eval()
        output = self._evalute(input)
        logging.info('Evaluation {} has finished'.format(self.name))
        return output