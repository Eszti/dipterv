import logging

from steps.eval.eval import Evaluation


class LangWiseEvaluation(Evaluation):
    def get_header(self):
        raise NotImplementedError

    def get_row_values(self, lang, list, univ, univ_cos):
        raise NotImplementedError

    def _evalute(self):
        input = self.input
        headers = self.get_header()
        logging.debug('Headers: {}'.format(headers))
        data = input[0]
        univ = input[1]
        univ_cos = self._get_cos_sim_mx(univ)
        results = []
        results.append(headers)
        for lang, list in data.iteritems():
            logging.info('Processing {}'.format(lang.upper()))
            row = []
            row.append(lang)
            row += self.get_row_values(lang, list, univ, univ_cos)
            results.append(row)
            logging.debug(row)
        return results