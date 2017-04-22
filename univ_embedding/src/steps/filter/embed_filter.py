import logging

from helpers import find_all_indices
from steps.filter.filter import Filter

# input : lang : swad_list, raw_emb_list, emb_fn, not_found_list
# output : lang : swad_list, raw_emb_list, emb_fn, not_found_list

class EmbedFilter(Filter):
    def _get_output_desc(self):
        return 'input : lang : swad_list, raw_emb_list, emb_fn, not_found_list\n' \
               'output : lang : swad_list, raw_emb_list, emb_fn, not_found_list'

    def init_filter(self):
        self.cutoff = self.get('cutoff', 'int')

    def filter(self, input):
        output = dict()
        for lang, list in input.iteritems():
            emb_list = list[1]
            emb_valid_len = len(emb_list) - len(find_all_indices(emb_list, None))
            if emb_valid_len >= self.cutoff:
                output[lang] = list
            else:
                logging.info('Removing language: {0} - {1} < {2}'.format(lang, emb_valid_len, self.cutoff))
        logging.info('Number of remaining languages: {}'.format(len(output)))
        return output