import csv

import os

from basic_eval import BasicEval
from process import Process

# input : ( [lang : swad_list, emb_full (norm), not_found_list, T], univ(norm))

class EvaluationProcess(Process):
    def save_output(self, output):
        pass

    def save_eval(self, filename, data):
        with open(filename, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(data)

    def init_for_do(self):
        eval_strs = self.get('evals').split('|')
        self.evals = []
        for eval_str in eval_strs:
            if eval_str == 'basic':
                self.evals.append(BasicEval)
        self.output_dir_name = os.path.join(self.output_dir, self.name)
        os.mkdir(self.output_dir_name)

    def _do(self):
        input = self.input
        for eval in self.evals:
            output = eval.evalute(input)
            filename = os.path.join(self.output_dir_name, '{}.csv'.format(self.name))
            self.save_eval(filename, output)
        return input