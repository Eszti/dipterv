import logging
import os

from steps.process.process import Process


class GetSwadProcess(Process):
# input : list of sil codes
# output : lang : swad_list_mul

    def __init__(self, name, genparams):
        super(GetSwadProcess, self).__init__(name, genparams)

    def _get_output_desc(self):
        return 'input : list of sil codes\n' \
               'output : lang : swad_list'

    def init_for_do(self):
        swad_root_dir = self.get('swad_root_dir')
        num = self.get('num', 'int')
        self.swad_dir = os.path.join(swad_root_dir, 'swadesh{}'.format(str(num)))

    def _read_swadesh(self, swad_fn):
        with open(swad_fn) as f:
            ls_swad = []
            lines = f.read().decode('utf-8').splitlines()
            for (i, line) in enumerate(lines):
                if line == '':
                    words = None
                else:
                    words = line.split('\t')
                ls_swad.append(words)
        return ls_swad

    def _do(self):
        output = dict()
        langs = self.input
        for lang in langs:
            swad_fns = [filename for filename in os.listdir(self.swad_dir) if filename.startswith(lang)]
            logging.info('{0} number of swad lists are found {1}'.format(len(swad_fns), swad_fns))
            if len(swad_fns) > 0:
                swad_min = min(swad_fns)
                swad_fn = os.path.join(self.swad_dir, swad_min)
                logging.info('Swadesh list {} is used'.format(swad_fn))
                ls_swad = self._read_swadesh(swad_fn)
            else:
                logging.warning('Skipping language : {}'.format(lang.upper()))
                continue
            output[lang] = ls_swad
        logging.info('{} languages are found'.format(len(output)))
        return output