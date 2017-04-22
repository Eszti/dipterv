import logging

import numpy as np
import os
from sklearn.preprocessing import normalize

from helpers import create_timestamped_dir, save_nparr, get_rowwise_norm
from process import Process
from train import train

# input : lang : swad_list, emb_full (norm), not_found_list
# output : ( [lang : swad_list, emb_full (norm), not_found_list, T], univ(norm))

class FindUnivProcess(Process):
    def _get_output_desc(self):
        raise NotImplementedError

    def init_for_do(self):
        self.trans_output_dir = self.get('output_dir')
        self.num_steps = self.get('num_steps', 'int')
        self.learning_rate = self.get('learning_rate', 'float')
        self.end_cond = self.get('end_cond', 'float')
        self.max_iter = self.get('max_iter', 'int')
        self.loss_crit = self.get('loss_crit', 'float')
        self.loss_crit_flag = self.get('loss_crit_flag', 'boolean')

    def _do(self):
        ts_output_dir = create_timestamped_dir(os.path.join(self.trans_output_dir))
        save_output_dir = os.path.join(ts_output_dir, 'saved')
        os.makedirs(save_output_dir)
        train_output = os.path.join(ts_output_dir, 'train_log')
        os.makedirs(train_output)
        input = self.input
        output = input
        eng_emb = input['eng'][1]
        W = np.ndarray(shape=(len(input), eng_emb.shape[0], eng_emb.shape[1]), dtype=np.float32)
        W[0, :, :] = eng_emb
        logging.info('Embedding {0} is in position {1}'.format('eng', 0))
        lang_pos = dict()
        i = 1
        for lang, list in input.iteritems():
            if lang == 'eng':
                continue
            logging.info('Embedding {0} is in position {1}'.format(lang.upper(), i))
            lang_pos[lang] = i
            emb = list[1]
            W[i, :, :] = emb
            i += 1
        T1, T, A = train(W, num_steps=self.num_steps,
                         learning_rate=self.learning_rate,
                         output_dir=train_output,
                         loss_crit=self.loss_crit,
                         loss_crit_flag=self.loss_crit_flag,
                         end_cond=self.end_cond,
                         max_iter=self.max_iter)
        # Save output
        T1_fn = os.path.join(save_output_dir, 'T1.npy')
        T_fn = os.path.join(save_output_dir, 'T.npy')
        A_fn = os.path.join(save_output_dir, 'A.npy')
        save_nparr(T1_fn, T1)
        save_nparr(T_fn, T)
        save_nparr(A_fn, A)

        for lang, list in input.iteritems():
            if lang == 'eng':
                output[lang].append(T1)
            else:
                output[lang].append(T[lang_pos[lang]-1, :, :])
        # Normalzing universal embedding
        A_norm = normalize(A)
        row_norm = get_rowwise_norm(A_norm)
        logging.info('Universal embedding normalized, Frobenius norm: {0} embed len ({1})'
                     .format(row_norm, A_norm.shape[0]))
        diff = A_norm.shape[0] - row_norm
        if abs(diff) > 0.01:
            logging.warning('Something went wrong at normalizing, diff = {}'.format(diff))
        ret = (output, A)
        return ret