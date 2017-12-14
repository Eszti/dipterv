import json

import numpy as np
import os
from gensim.models import KeyedVectors

from calc_utils import calc_precision, read_word_pairs_tsv, wp_list_2_dict
from utils import print_verbose

langs = ['eng', 'ita']
dim = 300
T_fn = '/home/eszti/projects/dipterv/panlex/output/20171214_1744_22/T_1.npz'
precs_to_calc = [1, 3, 5]

# Get sil2fb dictionary
sil2fb_fn = '/home/eszti/projects/dipterv/notebooks/panlex/data/sil2fb.json'
with open(sil2fb_fn) as f:
     sil2fb = json.load(f)

# Read embeddings
fold = '/mnt/permanent/Language/Multi/FB'
# Dictionary for embeddings
d_models = dict()
for l in langs:
    fn = os.path.join(fold, 'wiki.{}'.format(sil2fb[l]), 'wiki.{}.vec'.format(sil2fb[l]))
    print_verbose('Reading embedding from {}'.format(fn))
    model = KeyedVectors.load_word2vec_format(fn, binary=False)
    model.syn0 /= np.sqrt((model.syn0**2).sum(1))[:, None]
    d_models[l] = model
    print_verbose('# word:\t{}'.format(len(model.syn0)))


# Word pairs data
fold = '/home/eszti/projects/dipterv/notebooks/panlex/smith/'
id1 = 0
id2 = 1
# Dict for word pairs
d_wps = dict()
done = set()
for lang1 in langs:
    for lang2 in langs:
        lang_pair = tuple(sorted([lang1, lang2]))
        if lang1 == lang2 or lang_pair in done:
            continue
        done.add(lang_pair)
        l1 = lang_pair[0]
        l2 = lang_pair[1]
        fn = os.path.join(fold, '{0}_{1}.tsv'.format(l1, l2))
        print_verbose('Reading word pair file: {0}'.format(fn))
        data = read_word_pairs_tsv(fn, id1, id2, False)
        d_wps[lang_pair] = data
        print_verbose('Number of word pairs found: {0}'.format(len(data)))

# Dict for dictionaries between each languages
d_dict = dict()
for ((l1, l2), wp_l) in d_wps.items():
    print_verbose('Creating dictionary for: {0}-{1}'.format(l1, l2))
    l12, l21 = wp_list_2_dict(wp_l)
    d_dict[(l1, l2)] = l12
    d_dict[(l2, l1)] = l21
    print_verbose('# word in: {0}-{1}:\t{2}'.format(l1, l2, len(l12)))
    print_verbose('# word in: {0}-{1}:\t{2}'.format(l2, l1, len(l21)))

# Dict for filtered models containing only the words used for training
d_tr_mods = dict()
for ((l1, l2), d) in d_dict.items():
    print_verbose('Reading {0}-{1} dictionary'.format(l1, l2))
    tr_mod = KeyedVectors()
    nf_list = []
    for i, w in enumerate(list(d.keys())):
        # Check if there's an embedding to the word
        if w not in d_models[l1]:
            nf_list.append(w)
    print_verbose('Words not found in embedding: {}'.format(nf_list))
    tr_mod.index2word = [x for x in list(d.keys()) if x not in nf_list]
    tr_mod.syn0 = np.ndarray(shape=(len(tr_mod.index2word), dim), dtype=np.float32)
    # Adding embedding to train model
    for i, w in enumerate(tr_mod.index2word):
        tr_mod.syn0[i, :] = d_models[l1][w]
    # Deleting not forund words from word pairs list
    change = False
    if l1 < l2:
        lang1 = l1
        lang2 = l2
    else:
        lang1 = l2
        lang2 = l1
        change = True
    d_wps[(lang1, lang2)] = [(w1, w2) for (w1, w2) in d_wps[(lang1, lang2)]
                             if not ((change and w2 in nf_list) or (not change and w1 in nf_list))]
    d_tr_mods[(l1, l2)] = tr_mod

# Logging new word pairs lists' length
print_verbose("Word pair lists' length")
for ((l1, l2), d) in d_wps.items():
    print_verbose('{0}-{1}\t# word pairs:\t{2}'.format(l1, l2, len(d)))

# Logging training models' length
print_verbose("Traing models' length")
for ((l1, l2), d) in d_wps.items():
    print_verbose('{0}-{1}\t# word pairs:\t{2}'.format(l1, l2, len(d)))

# Read T mx from file
with open(T_fn) as f:
    nzpf = np.load(f)
    T = nzpf['T']
T1 = T[0]
T2 = T[1]

# Calculate precision
l1 = 'eng'
l2 = 'ita'
m1_tr = d_tr_mods[(l1, l2)]
m2_tr = d_tr_mods[(l2, l1)]
m1 = d_models[l1]
m2 = d_models[l2]
# Prec l1 - l2 = eng - ita
m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
m2.syn0 = np.dot(m2.syn0, T2)
precs_1 = calc_precision(precs_to_calc, m1_tr, m2, d_dict[(l1, l2)], verbose=False)
print_verbose('Precs: {0}-{1}\t{2}'.format(l1, l2, precs_1))
# Prec l2 - l1 = ita - eng
m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
m1.syn0 = np.dot(m1.syn0, T1)
precs_2 = calc_precision(precs_to_calc, m2_tr, m1, d_dict[(l2, l1)], verbose=False)
print_verbose('Precs: {0}-{1}\t{2}'.format(l2, l1, precs_2))