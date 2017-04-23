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
        self.save_if_skip = self.get('save_if_skip', section='skip', type='boolean')

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

    def save_output(self, output):
        step_output_dir_name = os.path.join(self.output_dir, self.name)
        os.mkdir(step_output_dir_name)
        filename = os.path.join(step_output_dir_name, '{}.pickle'.format(self.name))
        with open(filename, 'w') as f:
            pickle.dump(output, f)
        logging.info('Output has been saved to {}'.format(filename))

    def _get_output_desc(self):
        return ''

    def create_output_descriptor(self):
        filename = os.path.join(self.output_dir, self.name, 'README')
        with open(filename, 'w') as f:
            desc = self._get_output_desc()
            f.writelines(desc)
        logging.info('Descriptor file has been saved to {}'.format(filename))

    def run(self, input, do=False):
        logging.info('Starting step {}...'.format(self.name.upper()))
        output = self._run(input, do)
        if do or self.save_if_skip:
            logging.info('Saving output at the end of the step')
            self.save_output(output)
            self.create_output_descriptor()
        else:
            logging.info('Skipping saving output at the end of the step')
        logging.info('Finishing step {}...'.format(self.name.upper()))
        return output

    def _run(self, input, do=False):
        raise NotImplementedError