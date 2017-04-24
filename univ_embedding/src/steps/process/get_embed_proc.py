import json
import logging

import os

from helpers import find_all_indices
from steps.process.process import Process


class GetEmbedProcess(Process):
# input : lang : swad_list
# output : lang : swad_list, raw_emb_list, emb_fn, not_found_list

    def _get_output_desc(self):
        return 'input : lang : swad_list\n' \
               'output : lang : swad_list, raw_emb_list, emb_fn, not_found_list'

    def init_for_do(self):
        self.emb_dir = self.get('emb_dir')
        sil_to_fb_fn = self.get( 'sil_to_fb')
        with open(sil_to_fb_fn) as f:
            self.sil_to_fb = json.load(f)
        logging.info('{0} languages are found in {1} sil-to-fb mapping file'
                     .format(len(self.sil_to_fb), sil_to_fb_fn))

    def _do(self):
        output = self.input
        for sil, swad_list in self.input.iteritems():
            swad_valid_len = len(swad_list) - len(find_all_indices(swad_list, None))
            emb_list = [None] * len(swad_list)
            embed_fn = os.path.join(self.emb_dir, 'wiki.{0}/wiki.{0}.vec'.format(self.sil_to_fb[sil]))
            logging.info('Loading embedding {0} from {1}'.format(sil, embed_fn))
            logging.info('Valid swadesh length: {}'.format(swad_valid_len))
            with open(embed_fn) as f:
                idxs_found = []
                i = 0
                for line in f:
                    if i == 0:
                        i += 1
                        continue
                    fields = line.strip().decode('utf-8').split(' ')
                    w = fields[0]
                    w = w.lower()
                    idxs = find_all_indices(swad_list, w)
                    if len(idxs) == 0:
                        continue
                    for idx in idxs:
                        emb_list[idx] = fields[1:]
                    idxs_found += idxs
                    if len(idxs_found) == swad_valid_len:
                        break
            not_found_list = find_all_indices(emb_list, None)
            emb_valid_len = len(emb_list) - len(not_found_list)
            logging.info('Valid embedding len: {}'.format(emb_valid_len))
            logging.info('Not found: {}'.format(len(swad_list) - emb_valid_len))
            output[sil] = []
            output[sil].append(swad_list)
            output[sil].append(emb_list)
            output[sil].append(embed_fn)
            output[sil].append(not_found_list)
        return output