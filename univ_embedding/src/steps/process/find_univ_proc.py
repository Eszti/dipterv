import logging

import numpy as np
import os
from sklearn.preprocessing import normalize

from helpers import create_timestamped_dir, save_nparr, get_rowwise_norm, load_nparr
from process import Process
from steps.train import train

# input : lang : swad_list, emb_full (norm), emb_fn, not_found_list
# output : ( [lang : swad_list, emb_full (norm), emb_fn, not_found_list, T], univ(norm))

# pre: TranslateEmbProcess
class FindUnivProcess(Process):
    def _get_output_desc(self):
        return 'input : lang : swad_list, emb_full (norm), emb_fn, not_found_list\n' \
               'output : ( [lang : swad_list, emb_full (norm), not_found_list, T], univ(norm))'

    def init_for_do(self):
        self.trans_output_dir = self.get('output_dir')
        self.num_steps = self.get('num_steps', 'int')
        self.learning_rate = self.get('learning_rate', 'float')
        self.end_cond = self.get('end_cond', 'float')
        self.max_iter = self.get('max_iter', 'int')
        self.loss_crit = self.get('loss_crit', 'float')
        self.loss_crit_flag = self.get('loss_crit_flag', 'boolean')
        self.cont = self.get('continue', 'boolean')
        self.dir_for_initials = self.get('dir_for_initials')

    def _init_for_cont(self):
        dir = self.dir_for_initials
        filenames = [fn for fn in os.listdir(dir)]
        step_str = filenames[0].split('_')[1].split('.')[0]
        step = int(step_str)
        logging.info('Step is set to {}'.format(step))
        T1_path = os.path.join(dir, 'T1_{}.npy'.format(step))
        logging.info('Loading T1 from {}'.format(T1_path))
        T1_init = load_nparr(T1_path)
        T_path = os.path.join(dir, 'T_{}.npy'.format(step))
        logging.info('Loading T from {}'.format(T_path))
        T_init = load_nparr(T_path)
        A_path = os.path.join(dir, 'A_{}.npy'.format(step))
        logging.info('Loading A from {}'.format(A_path))
        A_init = load_nparr(A_path)
        skip_input_dir = self.get('load_input_dir', section='load')
        log_fn = os.path.join(skip_input_dir, 'log.txt')
        logging.info('Reconstruct language order from log file: {}'.format(log_fn))
        lang_list = []
        with open(log_fn) as f:
            lines = f.readlines()
            for line in lines:
                if 'is in position' in  line:
                    sil = line.split(' ')[3].lower()
                    lang_list.append(sil)
        logging.debug(lang_list)
        return step, T1_init, T_init, A_init, lang_list

    def _do(self):
        ts_output_dir = create_timestamped_dir(os.path.join(self.trans_output_dir))
        save_output_dir = os.path.join(ts_output_dir, 'saved')
        os.makedirs(save_output_dir)
        train_output = os.path.join(ts_output_dir, 'train_log')
        os.makedirs(train_output)
        input = self.input
        output = input
        # Whether we should continue
        T1_init, T_init, A_init = None, None, None
        if self.cont:
            logging.info('Continue training!')
            step, T1_init, T_init_raw, A_init, lang_list = self._init_for_cont()
            T_init = np.ndarray(shape=(T_init_raw.shape[0], T_init_raw.shape[1], T_init_raw.shape[2]), dtype=np.float32)
        else:
            T1_init, T_init, A_init = None, None, None
            step = 0
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
            if self.cont:
                T_init[i-1, :, :] = T_init_raw[lang_list.index(lang) - 1, :, :]
            i += 1

        T1, T, A = train(W, num_steps=self.num_steps,
                         learning_rate=self.learning_rate,
                         output_dir=train_output,
                         loss_crit=self.loss_crit,
                         loss_crit_flag=self.loss_crit_flag,
                         end_cond=self.end_cond,
                         max_iter=self.max_iter,
                         T_initial=T_init,
                         T1_initial=T1_init,
                         A_initial=A_init,
                         step_initial=step,
                         verbose=True)
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