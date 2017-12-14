import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import print_debug, print_verbose

# Read word pairs from tsv
def read_word_pairs_tsv(fn, id1, id2, header=True):
    with open(fn) as f:
        lines = f.readlines()
        data = [(line.split()[id1], line.split()[id2]) for i, line in enumerate(lines) if i > 0 or header == False]
    return data

# Get dictionary fromm wordlist
def wp_list_2_dict(wp_l):
    l12 = dict()
    l21 = dict()
    for (w1, w2) in wp_l:
        if w1 not in l12:
            l12[w1] = [w2]
        else:
            l12[w1].append(w2)
        if w2 not in l21:
            l21[w2] = [w1]
        else:
            l21[w2].append(w1)
    return l12, l21

# Function to calculate precision
# model_src : source language embeddings (need to have syn0 and index2word properites) (after translation)
# model_tar : target language embeddings (need to have syn0 and index2word properites) (can be don in orig or universal space)
# dict_scr_2_tar : dictionary from source to target
def calc_precision(precs, model_src, model_tar, dict_scr_2_tar, verbose=False):
    W_src = model_src.syn0
    W_tar = model_tar.syn0
    idx_src = model_src.index2word
    idx_tar = model_tar.index2word

    cos_mx = cosine_similarity(W_src, W_tar)
    sim_mx = np.argsort(-cos_mx)
    max_prec = max(precs)
    prec_cnt = np.zeros(shape=(1, max_prec))
    if verbose:
        print_debug('word: \ttranslations in dict: \tclosest words after translation: \t')
    for i, r in enumerate(sim_mx):
        key_word = idx_src[i]
        value_words = dict_scr_2_tar[key_word]
        closest_words = []
        for j in range(max_prec):
            ans = np.where(r == j)
            idx_orig = ans[0][0]
            word = idx_tar[idx_orig]
            closest_words.append(word)
            if word in value_words:
                prec_cnt[0][j] = prec_cnt[0][j] + 1
        if verbose:
            print_verbose('{}"\t{}\t{}'.format(key_word, value_words, closest_words))
    if verbose:
        print_verbose(prec_cnt)
    prec_pcnts = []
    for i, val in enumerate(precs):
        sum_hit = np.sum(prec_cnt[0][0:val])
        pcnt = float(sum_hit) / sim_mx.shape[0]
        if verbose:
            print_verbose('prec {} : {}'.format(val, pcnt))
        prec_pcnts.append(pcnt)
    return prec_pcnts