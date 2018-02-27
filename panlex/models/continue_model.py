import sys

import os
import re

sys.path.insert(0, 'utils')
from io_helper import load_pickle
import strings


from base.loggable import Loggable


class ContModel(Loggable):
    def __init__(self, cont_config):
        Loggable.__init__(self)
        self.cont = cont_config.cont
        if self.cont:
            self.logger.info('Continuing training...')
            input_folder = os.path.join(cont_config.input_folder, strings.TRAIN_OUTPUT_FOLDER_NAME)
            epoch = cont_config.epoch
            if epoch is None:
                list_of_files = os.listdir(input_folder)
                max_num = 0
                if epoch is None:
                    for file in list_of_files:
                        s = re.search('T_(\d*).pickle', file)   # assuming filename is "T_x.pickle"
                        if s is not None:
                            num = int(s.group(1))
                            max_num = num if num > max_num else max_num
                    epoch = max_num
            T_fn = os.path.join(input_folder, 'T_{}.pickle'.format(epoch))
            self.logger.info('Loading transformation matrix from {}'.format(T_fn))
            self.T_loaded = load_pickle(T_fn)