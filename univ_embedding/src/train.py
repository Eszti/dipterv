from __future__ import print_function
import numpy as np
import utils, json, os, sys
from os import listdir
from os.path import isfile, join
import logging

def _train_univ_embed(eng_fn, filenames, num_steps, verbose, output_dir=None):
    lang_cnt = len(filenames)

    # Load English embedding
    en_emb = utils.load_nparr(eng_fn)

    W = np.ndarray(shape=(lang_cnt, en_emb.shape[0], en_emb.shape[1]), dtype=np.float32)
    W[0, :, :] = en_emb
    i = 1
    for fn in filenames:
        if 'eng_eng' in fn:
            continue
        emb = utils.load_nparr(fn)
        W[i, :, :] = emb
        i += 1
    T1, T, A = utils.train(W, num_steps=num_steps, verbose=verbose, output_dir=output_dir)
    return T1, T, A


def train_univ_embed(cfg):
    # Config values
    eng_fn = cfg.get('general', 'eng_fn')
    silcodes_fn = cfg.get('general', 'silcodes_fn')
    embed_dir = cfg.get('general', 'embed_dir')
    output = cfg.get('general', 'output')
    num_steps = cfg.getint('train', 'num_steps')
    verbose = cfg.getboolean('train', 'verbose')
    # Logging
    logging.info('English embedding file: {}'.format(eng_fn))
    logging.info('Silcodes file: {}'.format(silcodes_fn))
    logging.info('Embedding dir: {}'.format(embed_dir))
    logging.info('Output folder: {}'.format(output))
    logging.info('Number of steps: {}'.format(num_steps))
    logging.info('Verbose: {}'.format(verbose))

    # Load silcodes
    with open(silcodes_fn) as f:
        silcodes = json.load(f)
    lang_cnt = len(silcodes)

    # Get embedding filenames
    embed_files = [os.path.join(embed_dir, f)
                   for f in listdir(embed_dir) if isfile(join(embed_dir, f)) and f.endswith('.npy')]
    embed_cnt = len(embed_files)
    if lang_cnt != embed_cnt:
        logging.warning('sil codes and embedding not equal: {0} != {1} number of embeddings are counted'
                        .format(lang_cnt, embed_cnt))

    output_dir = utils.create_timestamped_dir(output)
    # Train
    T1, T, A = _train_univ_embed(eng_fn, embed_files, num_steps, verbose, output_dir)
    # Save output
    T1_fn = os.path.join(output_dir, 'T1.npy')
    T_fn = os.path.join(output_dir, 'T.npy')
    A_fn = os.path.join(output_dir, 'A.npy')
    utils.save_nparr(T1_fn, T1)
    utils.save_nparr(T_fn, T)
    utils.save_nparr(A_fn, A)


def test():
    W1 = np.array([[1, 0], [0, 1], [1, 1]]).astype(np.float32)
    W2 = np.array([[2, 2], [1, -1], [-1, -1]]).astype(np.float32)
    W = np.ndarray(shape=(2, 3, 2), dtype=np.float32)
    W[0, :, :] = W1
    W[1, :, :] = W2
    T1, T, A = utils.train(W, learning_rate=1)

def main():
    os.nice(20)
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = utils.get_cfg(cfg_file)
    train_univ_embed(cfg)

if __name__ == '__main__':
    main()