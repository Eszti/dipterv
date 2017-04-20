from __future__ import print_function

import numpy as np
from os import listdir
from os.path import isfile, join

import json
import logging
import os
import sys
import time
import utils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s '
                           '%(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

def _train_univ_embed(embed_filenames, cfg_train, starttime):
    lang_cnt = len(embed_filenames)

    # Load English embedding
    en_emb = utils.load_nparr(cfg_train.eng_emb_fn)

    W = np.ndarray(shape=(lang_cnt, en_emb.shape[0], en_emb.shape[1]), dtype=np.float32)
    W[0, :, :] = en_emb
    logging.info('Embedding {0} is in position {1}'.format(cfg_train.eng_emb_fn, 0))
    i = 1
    for fn in embed_filenames:
        if 'eng_eng' in fn:
            continue
        logging.info('Embedding {0} is in position {1}'.format(fn, i))
        emb = utils.load_nparr(fn)
        W[i, :, :] = emb
        i += 1
    T1, T, A = utils.train(W, starttime, num_steps=cfg_train.num_steps, verbose=cfg_train.verbose,
                           output_dir=cfg_train.output_dir, log_freq=cfg_train.log_freq,
                           end_cond=cfg_train.end_cond, max_iter=cfg_train.max_iter, debug=cfg_train.debug)
    return T1, T, A


def train_univ_embed(cfg, startime):
    # Config values
    cfg_train = utils.get_train_config(cfg)
    # Logging
    # General
    logging.info('Output folder: {}'.format(cfg_train.output_dir))
    # Codes
    logging.info('Silcodes file: {}'.format(cfg_train.silcodes_fn))
    # Embedding
    logging.info('Embedding dir: {}'.format(cfg_train.emb_dir))
    logging.info('English embedding file: {}'.format(cfg_train.eng_emb_fn))
    # Swadesh
    logging.info('Swadesh dir: {}'.format(cfg_train.swad_dir))
    # Train
    logging.info('Verbose: {}'.format(cfg_train.verbose))
    logging.info('Logging frequency: {}'.format(cfg_train.log_freq))
    logging.info('Number of steps: {}'.format(cfg_train.num_steps))
    logging.info('End condition: {}'.format(cfg_train.end_cond))
    logging.info('Learning rate: {}'.format(cfg_train.learning_rate))
    logging.info('Max iter: {}'.format(cfg_train.max_iter))

    # Load silcodes
    logging.info('Reading silcodes...')
    with open(cfg_train.silcodes_fn) as f:
        silcodes = json.load(f)
    sil_cnt = len(silcodes)
    logging.info('Number of sil codes: {}'.format(sil_cnt))

    # Get embedding filenames
    logging.info('Reading embeddings...')
    embed_files = [os.path.join(cfg_train.emb_dir, f)
                   for f in listdir(cfg_train.emb_dir) if isfile(join(cfg_train.emb_dir, f)) and f.endswith('.npy')]
    embed_cnt = len(embed_files)
    logging.info('Number of embeddings: {}'.format(embed_cnt))
    if sil_cnt != embed_cnt:
        logging.warning('sil codes and embedding not equal: {0} != {1} number of embeddings are taken into account'
                        .format(sil_cnt, embed_cnt))

    output_dir = utils.create_timestamped_dir(cfg_train.output_dir)
    cfg_train.output_dir = output_dir
    # Train
    T1, T, A = _train_univ_embed(embed_files, cfg_train, startime)
    # Save output
    T1_fn = os.path.join(output_dir, 'T1.npy')
    T_fn = os.path.join(output_dir, 'T.npy')
    A_fn = os.path.join(output_dir, 'A.npy')
    utils.save_nparr(T1_fn, T1)
    utils.save_nparr(T_fn, T)
    utils.save_nparr(A_fn, A)
    logging.info('Training is finished, A, T1, T are saved!')

def test():
    W1 = np.array([[1, 0], [0, 1], [1, 1]]).astype(np.float32)
    W2 = np.array([[2, 2], [1, -1], [-1, -1]]).astype(np.float32)
    W = np.ndarray(shape=(2, 3, 2), dtype=np.float32)
    W[0, :, :] = W1
    W[1, :, :] = W2
    T1, T, A = utils.train(W, learning_rate=1)

def main():
    starttime = int(round(time.time()))
    os.nice(20)
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    logging.info('Config file: {}'.format(cfg_file))
    cfg = utils.get_cfg(cfg_file)
    train_univ_embed(cfg, starttime)
    finishtime = int(round(time.time()))
    logging.info('Running time in seconds: {}'.format(finishtime - starttime))

if __name__ == '__main__':
    main()