import csv

import os

from data_structures import GeneralParams
from process import Process
from steps.eval.basic_eval import BasicEvaluation
from steps.eval.swad_eval import SwadeshEvaluation
from steps.eval.top_n_eval import TopNEvaluation


class EvaluationProcess(Process):
# input : ( [lang : swad_list, emb_full (norm), emb_fn, not_found_list, T], univ(norm))
    def save_output(self, output):
        pass

    def save_eval(self, filename, data):
        with open(filename, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(data)

    def init_for_do(self):
        genparams = GeneralParams(self.starttime, self.config, self.output_dir)
        eval_strs = self.get('evals').split('|')
        self.evals = []
        for eval_str in eval_strs:
            if eval_str == 'basic':
                name = os.path.join(self.name, 'basic_eval')
                self.evals.append(BasicEvaluation(name, genparams))
            if eval_str == 'top_n':
                name = os.path.join(self.name, 'top_n_eval')
                self.evals.append(TopNEvaluation(name, genparams))
            if eval_str == 'swad':
                name = os.path.join(self.name, 'swad_eval')
                self.evals.append(SwadeshEvaluation(name, genparams))
        self.output_dir_name = os.path.join(self.output_dir, self.name)
        os.mkdir(self.output_dir_name)

    def _do(self):
        input = self.input
        for eval in self.evals:
            output = eval.evalute(input)
            filename = os.path.join(self.output_dir_name, '{}.csv'.format(eval.fn))
            self.save_eval(filename, output)
        return input