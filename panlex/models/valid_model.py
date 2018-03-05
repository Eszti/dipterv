import sys

sys.path.insert(0, 'utils')
sys.path.insert(0, 'models')

from test_base_model import TestBaseModel

import strings


class ValidModel(TestBaseModel):
    def __init__(self, valid_config, data_model_wrapper, language_config, output_dir):
        TestBaseModel.__init__(self, model_config=valid_config, data_model_wrapper=data_model_wrapper,
                               language_config=language_config, output_dir=output_dir, type=strings.VALID)

    def do_validation(self, svd_done, epoch, T):
        valid_done = False
        if svd_done or epoch % self.model_config.do_on == 0:
            self.do(epoch=epoch, T=T)
            valid_done = True
        return valid_done