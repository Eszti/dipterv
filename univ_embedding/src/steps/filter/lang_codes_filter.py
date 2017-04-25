import json

import logging

from steps.filter.filter import Filter

# input : list of sil codes
# output : list of sil codes

class LangCodesFilter(Filter):
    def _get_output_desc(self):
        return 'input: list of sil codes\n' \
               'output: list of sil codes'

    def init_filter(self):
        codes_fn = self.get('codes_to_retain')
        with open(codes_fn) as f:
            self.codes_to_retain = json.load(f)

    def filter(self, input):
        output = dict()
        for lang, val in input.iteritems():
            if lang in self.codes_to_retain:
                output[lang] = val
            else:
                logging.info('{} is removed'.format(lang))
        logging.info('{0} remained form {1}'.format(len(output), len(input)))
        return output