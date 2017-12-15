import os

from base.loggable import Loggable


class DataModel(Loggable):
    def __init__(self, language_config, data_config):
        Loggable.__init__(self)
        self.word_pairs_dict = dict()
        self.language_config = language_config
        self.data_config = data_config

    def _get_word_pairs_dict(self):
        done = set()
        for lang1 in self.language_config.langs:
            for lang2 in self.language_config.langs:
                lang_pair = tuple(sorted([lang1, lang2]))
                if lang1 == lang2 or lang_pair in done:
                    continue
                done.add(lang_pair)
                l1 = lang_pair[0]
                l2 = lang_pair[1]
                fn = os.path.join(self.data_config.dir, '{0}_{1}.tsv'.format(l1, l2))
                self.logger.info('Reading word pair file: {0}'.format(fn))
                data = self._read_word_pairs_tsv(fn, self.data_config.idx1, self.data_config.idx2, False)
                self.word_pairs_dict[lang_pair] = data
                self.logger.info('Number of word pairs found: {0}'.format(len(data)))

    # Read word pairs from tsv
    def _read_word_pairs_tsv(self, fn, id1, id2, header=True):
        with open(fn) as f:
            lines = f.readlines()
            data = [(line.split()[id1], line.split()[id2]) for i, line in enumerate(lines) if i > 0 or header == False]
        return data

    # Get dictionary fromm wordlist
    def _wp_list_2_dict(self, wp_l):
        l12 = dict()
        l21 = dict()
        for (w1, w2) in wp_l:
            if w1 not in l12:
                l12[w1] = [w2]
            else:
                l12[w1].append(w2)
            if w2 not in l21:
                l21[w2] = [w1]
            else:
                l21[w2].append(w1)
        return l12, l21

class DataModelWrapper(Loggable):
    def __init__(self):
        Loggable.__init__(self)