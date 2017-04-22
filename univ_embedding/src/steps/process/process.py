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
            return self.skip()

    def init_for_do(self):
        raise NotImplementedError

    def init_for_skip(self):
        skip_root = self.get('skip_input_dir', section='skip')
        self.skip_fn = os.path.join(skip_root, self.name, '{}.pickle'.format(self.name))

    def _do(self):
        raise NotImplementedError

    def _skip(self):
        logging.info('Loading input from: {}'.format(self.skip_fn))
        with open(self.skip_fn) as f:
            output = pickle.load(f)
        return output

    def do(self):
        logging.info('Do function is called')
        self.init_for_do()
        return self._do()

    def skip(self):
        logging.info('Skip function is called')
        self.init_for_skip()
        return self._skip()