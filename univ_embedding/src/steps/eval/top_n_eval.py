import logging

import numpy as np
import scipy.stats

from steps.eval.langwise_eval import LangWiseEvaluation


class TopNEvaluation(LangWiseEvaluation):
# input : ( [lang : swad_list, emb_full (norm), emb_fn, not_found_list, T], univ(norm))

    def _init_for_eval(self):
        self.top = self.get('top', 'int')

    def get_header(self):
        headers = ['lang_code',
                   'cos(orig, trans)'
                    ]
        return headers

    def get_row_values(self, lang, list, univ, univ_cos):
        row = []
        embed_fn = list[2]
        T = list[4]
        emb_list = []
        logging.info('Loading embedding from {}'.format(embed_fn))
        with open(embed_fn) as f:
            i = 0
            for line in f:
                if i == 0:
                    i += 1
                    continue
                fields = line.strip().decode('utf-8').split(' ')
                emb_list.append(fields[1:])
                if i == self.top:
                    break
                i += 1
        if i != self.top:
            logging.warning('{0} language does not have {1} of entries in {2}'
                            .format(lang.upper(), self.top, embed_fn))
        logging.info('Loaded embedding length: {}'.format(len(emb_list)))
        emb = np.array(emb_list).astype(np.float32)
        trans = np.dot(emb, T)
        pears = self.cos_corr(emb, trans)
        row.append(pears)
        return row