import logging
import os

from steps.process.process import Process


class GetSwadProcess(Process):
    def init_for_do(self):
        section = self.name
        swad_root_dir = self.config.get(section, 'swad_root_dir')
        num = self.config.getint(section, 'num')
        self.swad_dir = os.path.join(swad_root_dir, 'swadesh{}'.format(str(num)))

    def init_for_skip(self):
        raise NotImplementedError

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
            swad_fn = os.path.join(self.swad_dir, '{}-000.txt'.format(lang))
            try:
                ls_swad = self._read_swadesh(swad_fn)
            except:
                logging.warning('{0} does not exist'.format(swad_fn))
                try:
                    swad_fn2 = swad_fn.replace('000', '001')
                    ls_swad = self._read_swadesh(swad_fn2)
                    logging.warning('{0} is used'.format(swad_fn2))
                except:
                    logging.warning('{0} does not exist EITHER'.format(swad_fn2))
                    raise Exception('NOSwadesh')
            output[lang] = ls_swad
        return output

    def _skip(self):
        # TODO
        return self.do()