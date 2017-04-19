import pickle

import logging
import os

__metaclass__ = type
class Step():
    def __init__(self, name, genparams):
        self.name = name
        self.config = genparams.config
        self.starttime = genparams.starttime
        self.output_dir = genparams.output_dir

    def save_output(self, output):
        step_output_dir_name = os.path.join(self.output_dir, self.name)
        os.mkdir(step_output_dir_name)
        filename = os.path.join(step_output_dir_name, '{}.pickle'.format(self.name))
        with open(filename, 'w') as f:
            pickle.dump(output, f)

    def _get_output_desc(self):
        raise NotImplementedError

    def create_output_descriptor(self):
        filename = os.path.join(self.output_dir, self.name, 'README')
        with open(filename, 'w') as f:
            desc = self._get_output_desc()
            f.writelines(desc)

    def run(self, input, do=False):
        logging.info('Starting step {}...'.format(self.name))
        output = self._run(input, do)
        self.save_output(output)
        self.create_output_descriptor()
        logging.info('Finishing step {}...'.format(self.name))
        return output

    def _run(self, input, do=False):
        raise NotImplementedError