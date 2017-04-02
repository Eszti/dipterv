from __future__ import print_function
import numpy as np
import utils, json, os
from os import listdir
from os.path import isfile, join
import logging


def _load_nparr(embed_fn):
    with open(embed_fn) as f:
        emb = np.load(f)
    return emb

def _save_nparr(embed_fn, emb):
    with open(embed_fn, 'w') as f:
        np.safe_eval(f, emb)

def _train_univ_embed(eng_fn, filenames):
    lang_cnt = len(filenames)

    # Load English embedding
    en_emb = _load_nparr(eng_fn)

    W = np.ndarray(shape=(lang_cnt, en_emb.shape[0], en_emb.shape[1]), dtype=np.float32)
    W[0, :, :] = en_emb
    i = 1
    for fn in filenames:
        if 'eng_eng' in fn:
            continue
        emb = _load_nparr(fn)
        W[i, :, :] = emb
        i += 1
    T1, T, A = utils.train(W, num_steps=500000, verbose=True)
    return T1, T, A


def train_univ_embed():
    # Config values
    eng_fn = '/home/eszti/data/embeddings/fb_trans/embedding/20170402_1806/eng_eng.npy'
    silcodes_fn = '/home/eszti/projects/dipterv/univ_embedding/res/swad_fb_110.json'
    embed_dir = '/home/eszti/data/embeddings/fb_trans/embedding/20170402_1806'
    output = '/home/eszti/projects/dipterv/univ_embedding/output'

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
    # Train
    T1, T, A = _train_univ_embed(eng_fn, embed_files)
    # Save output
    output_dir = utils.create_timestamped_dir(output)
    T1_fn = os.path.join(output_dir, 'T1.npy')
    T_fn = os.path.join(output_dir, 'T.npy')
    A_fn = os.path.join(output_dir, 'A.npy')
    _save_nparr(T1_fn, T1)
    _save_nparr(T_fn, T)
    _save_nparr(A_fn, A)


def test():
    W1 = np.array([[1, 0], [0, 1], [1, 1]]).astype(np.float32)
    W2 = np.array([[2, 2], [1, -1], [-1, -1]]).astype(np.float32)
    W = np.ndarray(shape=(2, 3, 2), dtype=np.float32)
    W[0, :, :] = W1
    W[1, :, :] = W2
    T1, T, A = utils.train(W, learning_rate=0.1)

def main():
    train_univ_embed()

if __name__ == '__main__':
    main()