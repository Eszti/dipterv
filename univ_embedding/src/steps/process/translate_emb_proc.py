import logging

import numpy as np
import os
from sklearn.preprocessing import normalize

from helpers import create_timestamped_dir, get_rowwise_norm, save_nparr, find_all_indices, filter_list
from steps.process.process import Process
from steps.train import train

class TranslateEmbProcess(Process):
    def _get_output_desc(self):
        desc = 'output = lang_swad_dict\n' \
               'lang_swad_dict = { lang_swad_entry }\n' \
               'lang_swad_entry = sil_code, value_list\n' \
               'value_list = swad_list, embed_list\n' \
               'swad_list = { word }\n' \
               'embed_list = { embedding }\n' \
               'word = ? all possible swadesh words ?\n' \
               'sil_code = ? all possible sil codes ?\n' \
               'embedding = ? all possible read word vectors completed by translation ?'
        return desc

    def init_for_do(self):
        self.save_output_flag = self.get('save_output', 'boolean')
        if self.save_output_flag:
            self.output_dir = self.get('output_dir')
        self.num_steps = self.get('num_steps', 'int')
        self.learning_rate = self.get('learning_rate', 'float')
        self.end_cond = self.get('end_cond', 'float')
        self.max_iter = self.get('max_iter', 'int')
        self.loss_crit = self.get('loss_crit', 'float')

    def init_for_skip(self):
        section = self.name
        self.input_dir = self.config.get(section, 'input_dir')

    def _do(self):
        if self.save_output_flag:
            output_dir = create_timestamped_dir(os.path.join(self.output_dir, 'saved'))
        input = self.input
        output = input
        eng_emb = np.array(input['eng'][1]).astype(np.float32)
        # Normalize English embedding
        eng_emb = normalize(eng_emb)
        for lang, list in input.iteritems():
            trans_list = []
            if lang == 'eng':
                output[lang][1] = eng_emb       # we have to save this as well
                continue
            train_output = os.path.join(self.output_dir, self.name, 'train_log', lang)
            emb_list = list[1]
            not_found_idxs = find_all_indices(emb_list, None)
            emb_list_filtered = filter_list(emb_list, not_found_idxs)
            # Normalize and filter embedding for translation
            emb_filtered = normalize(np.array(emb_list_filtered).astype(np.float32))
            # Filter English embedding for translation
            eng_emb_filtered = np.delete(eng_emb, not_found_idxs, 0)

            W = np.ndarray(shape=(2, emb_filtered.shape[0],
                                  emb_filtered.shape[1]), dtype=np.float32)
            W[0, :, :] = eng_emb_filtered
            W[1, :, :] = emb_filtered
            T1, T, A = train(W, num_steps=self.num_steps,
                             learning_rate=self.learning_rate,
                             output_dir=train_output,
                             loss_crit=self.loss_crit,
                             end_cond=self.end_cond, max_iter=self.max_iter)
            translation = np.dot(eng_emb, T[0])
            for i, entry in enumerate(emb_list):
                if entry is None:
                    trans_list.append(translation[i])
                else:
                    trans_list.append(entry)
            # Normalize translated embedding
            trans_list_norm = normalize(np.array(trans_list).astype(np.float32))
            row_norm = get_rowwise_norm(trans_list_norm)
            logging.info('Embedding normalized, Frobenius norm: {0} embed len ({1})'
                         .format(row_norm, trans_list_norm.shape[0]))
            diff = trans_list_norm.shape[0] - row_norm
            if abs(diff) > 0.01:
                logging.warning('Something went wrong at normalizing, diff = {}'.format(diff))
            output[lang][1] = trans_list_norm
            if self.save_output_flag:
                filename = os.path.join(output_dir, '{}.npy'.format(lang))
                save_nparr(filename, trans_list_norm)
                logging.info('Translation of {0} is saved into {1}'.format(lang, filename))
        return output

    def _skip(self):
        raise NotImplementedError