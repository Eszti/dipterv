import logging

import numpy as np
import os

from data_structures import GeneralParams
from helpers import retain_list
from steps.eval.langwise_eval import LangWiseEvaluation
from steps.filter.swad_filter import SwadFilter
from steps.process.get_embed_proc import GetEmbedProcess
from steps.process.get_swad_proc import GetSwadProcess

class SwadeshEvaluation(LangWiseEvaluation):
# input : ( [lang : swad_list, emb_full (norm), emb_fn, not_found_list, T], univ(norm))

    def _init_for_eval(self):
        genparams = GeneralParams(self.starttime, self.config, self.output_dir)
        load = self.get('load',  'boolean')
        do = True
        if load:
            do = False
        step_strs = self.get('steps').split('|')
        steps = []
        for step_str in step_strs:
            if step_str == 'get_swad_proc':
                name = os.path.join(self.name, 'get_swad_proc')
                steps.append(GetSwadProcess(name, genparams))
            if step_str == 'swad_filter':
                name = os.path.join(self.name, 'swad_filter')
                steps.append(SwadFilter(name, genparams))
            if step_str == 'get_embed_proc':
                name = os.path.join(self.name, 'get_embed_proc')
                steps.append(GetEmbedProcess(name, genparams))
        input = self.input[0].keys()
        for step in steps:
            output = step.run(input, do, load)
            input = output
        # output : lang : swad_list, raw_emb_list, emb_fn, not_found_list
        self.swad_for_eval = output

    def get_header(self):
        headers = ['lang_code',
                   'cos_swad(orig, trans)',
                   'train_on',
                   'eval_on'
                    ]
        return headers

    def get_row_values(self, lang, list, univ, univ_cos):
        row = []
        if lang not in self.swad_for_eval.keys():       # eval swad list does not exist
            logging.warning('Swad eval is not found')
            row.append(None)
        else:
            swad_list_orig = list[0]
            not_found_orig = list[3]
            swad_list_eval = self.swad_for_eval[lang][0]
            not_found_eval = self.swad_for_eval[lang][3]
            swad_orig_valid_len = len(swad_list_orig) - len(not_found_orig)
            swad_eval_valid_len = len(swad_list_eval) - len(not_found_eval)
            logging.info("Orig swad len: {0}\tOrig valid swad len: {1}".format(len(swad_list_orig), swad_orig_valid_len))
            logging.info("Eval swad len: {0}\tEval valid swad len: {1}".format(len(swad_list_eval), swad_eval_valid_len))
            eval_idxs = []
            for i, l in enumerate(swad_list_eval):
                if l is not None:
                    if l not in swad_list_orig and i not in not_found_eval:
                        eval_idxs.append(i)
            logging.info('Evaluate on {} occurences'.format(len(eval_idxs)))
            logging.debug('Evail indices: {}'.format(eval_idxs))
            T = list[4]
            emb_eval = np.array(retain_list(self.swad_for_eval[lang][1], eval_idxs)).astype(np.float32)
            logging.debug('Eval embed shape: {}'.format(emb_eval.shape))
            trans = np.dot(emb_eval, T)
            row.append(self.cos_corr(emb_eval, trans))
            row.append(swad_orig_valid_len)
            row.append(swad_eval_valid_len)
        return row