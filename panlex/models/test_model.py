import sys

import os

sys.path.insert(0, 'utils')
sys.path.insert(0, 'models')

from io_helper import load_pickle
from test_base_model import TestBaseModel
import strings


class TestModel(TestBaseModel):
    def __init__(self, test_config, data_model_wrapper, language_config, output_dir):
        TestBaseModel.__init__(self, model_config=test_config, data_model_wrapper=data_model_wrapper,
                               language_config=language_config, output_dir=output_dir, type=strings.TEST)

    def do_test(self):
        for epoch in self.model_config.epochs:
            T_fn = os.path.join(self.model_config.input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME,
                                'T_{}.pickle'.format(epoch))
            self.logger.info('Loading pickle from {}'.format(T_fn))
            T = load_pickle(T_fn)
            self.do(epoch=epoch, T=T)

    def run(self):
        self.do_test()