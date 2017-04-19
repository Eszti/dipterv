import logging

from steps.filter.filter import Filter
from steps.utils import find_all_indices


class EmbedFilter(Filter):
    def _get_output_desc(self):
        desc = 'output = lang_swad_dict\n' \
               'lang_swad_dict = { lang_swad_entry }\n' \
               'lang_swad_entry = sil_code, value_list\n' \
               'value_list = swad_list, embed_list\n' \
               'swad_list = { word }\n' \
               'embed_list = { embedding }\n' \
               'word = ? all possible swadesh words ?\n' \
               'sil_code = ? all possible sil codes ?\n' \
               'embedding = ? all possible read word vectors ?'
        return desc

    def init_filter(self):
        section = self.name
        self.cutoff = self.config.getint(section, 'cutoff')

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