import logging
import pickle

import os

from steps.step import Step


class Process(Step):
    def _run(self, input, do=False):
        self.input = input
        if do:
            return self.do()
        else:
            return self.load()

    def init_for_do(self):
        raise NotImplementedError

    def init_for_load(self):
        skip_root = self.get('load_input_dir', section='load')
        self.load_fn = os.path.join(skip_root, self.name, '{}.pickle'.format(self.fn))

    def _do(self):
        raise NotImplementedError

    def _load(self):
        logging.info('Loading input from: {}'.format(self.load_fn))
        with open(self.load_fn) as f:
            output = pickle.load(f)
        return output

    def do(self):
        logging.info('DO function is called')
        self.init_for_do()
        return self._do()

    def load(self):
        logging.info('LOAD function is called')
        self.init_for_load()
        return self._load()