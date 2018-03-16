import collections
import csv
import datetime
import pickle

import errno
import numpy as np

import os
from gensim.models import KeyedVectors

def checkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise OSError

debug = False

lang1 = 'eng'
lang2 = 'ita'

# Variables for reading tsv
working_dir = os.path.join('/mnt/permanent/home/eszti/dipterv/panlex/data/panlex/tsv', '{0}_{1}'.format(lang1, lang2))
full_tsv_fn = os.path.join(working_dir, '{0}_{1}.tsv'.format(lang1, lang2))
id1 = 2
id2 = 3
score_id = 4
lines = None

# Variables for splitting (eng - ita: 1-9)
score_min = 0
score_max = 9
tr_rat = 7

# Valiables for reading embeddings
lang1_emb_path = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'
lang2_emb_path = '/mnt/permanent/Language/Multi/FB/wiki.it/wiki.it.vec'
limit = 200000

# Read embeddings
print('Reading embeddings...')
print(lang1)
emb1 = KeyedVectors.load_word2vec_format(lang1_emb_path, binary=False, limit=limit)
print(lang2)
emb2 = KeyedVectors.load_word2vec_format(lang2_emb_path, binary=False, limit=limit)

# Read word pairs (w1, w2, score)
all_word_pairs = []
print('Reading tsv from {}'.format(full_tsv_fn))
with open(full_tsv_fn) as f:
    read_lines = f.readlines()
    for i, line in enumerate(read_lines):
        if lines is not None and i == lines:
            break
        fields = line.split('\t')
        try:
            all_word_pairs.append((fields[id1], fields[id2], float(fields[score_id])))
        except:
            print(i)
            print(line)
print('# all word pairs: {}'.format(len(all_word_pairs)))

gold_l1 = collections.defaultdict(set)
act_wps = []

for score in range(score_max, score_min-1, -1):
    print('\n')
    print(datetime.datetime.now())
    print('======== Score {} ========'.format(score))
    score_dir = os.path.join(working_dir, '{}'.format(score))

    # Filter word pair list (only score)
    act_score_wps = [(w1, w2) for (w1, w2, sc) in all_word_pairs if int(round(sc)) == score]
    fil_act_score_wps = [(w1, w2) for (w1, w2) in act_score_wps if w1 in emb1.index2word and w2 in emb2.index2word]
    act_wps += fil_act_score_wps
    print('# wp - score {0}: {1}'.format(score, len(act_score_wps)))
    print('# wp filtered - score {0}: {1}'.format(score, len(fil_act_score_wps)))
    print('# wp filtered - until score {0}: {1}'.format(score, len(act_wps)))

    # Create dictionary (append)
    print(datetime.datetime.now())
    print('Splitting...')
    for (w1, w2) in fil_act_score_wps:
        gold_l1[w1].add(w2)

    for (k, vals) in gold_l1.items():
        wp_s = []
        for v in vals:
            wp_s.append((k, v))

    # Split data (from the start)
    i = 0
    tr = []
    te = []
    for (k, vals) in gold_l1.items():
        wp_s = []
        for v in vals:
            wp_s.append((k, v))
        if i % 10 < tr_rat:
            tr += wp_s
        else:
            te += wp_s
        i += 1
    print('# train: {}'.format(len(tr)))
    print('# test: {}'.format(len(te)))

    # Save embeddings
    # All
    print('--------all--------')
    words1, words2 = zip(*act_wps)
    words1 = set(words1)
    words2 = set(words2)
    print('{}'.format(lang1))
    print('{0} len: {1}'.format(lang1, len(words1)))
    print('{0} len: {1}'.format(lang2, len(words2)))
    # Train
    print('--------train--------')
    tr_1_fn = os.path.join(score_dir, '{}.pickle'.format(lang1))
    tr_2_fn = os.path.join(score_dir, '{}.pickle'.format(lang2))
    tr_words1, tr_words2 = zip(*tr)
    tr_words1 = set(tr_words1)
    tr_words2 = set(tr_words2)
    print('{}'.format(lang1))
    print('{0} len: {1}'.format(lang1, len(tr_words1)))
    print('{0} len: {1}'.format(lang2, len(tr_words2)))
    # Test
    print('--------test--------')
    te_words1, te_words2 = zip(*te)
    te_words1 = set(te_words1)
    te_words2 = set(te_words2)
    print('{}'.format(lang1))
    print('{0} len: {1}'.format(lang1, len(te_words1)))
    print('{0} len: {1}'.format(lang2, len(te_words2)))

    # Train - Test overlap
    overlap_1 = set(tr_words1) & set(te_words1)
    overlap_2 = set(tr_words2) & set(te_words2)
    print('-----overlap test-----')
    print('{0} - {1}'.format(lang1, len(overlap_1)))
    print('{0} - {1}'.format(lang2, len(overlap_2)))

    # Save tsv
    print('Saving tsv-s...')
    act_tsv_fn = os.path.join(score_dir, '{0}_{1}.tsv'.format(lang1, lang2))
    tr_dir = os.path.join(score_dir, 'train')
    te_dir = os.path.join(score_dir, 'test')
    def save_tsv(dir, data):
        fn = os.path.join(dir, '{0}_{1}.tsv'.format(lang1, lang2))
        checkdir(fn)
        print('Saving tsv: {}'.format(fn))
        with open(fn, 'wt') as f:
            wr = csv.writer(f, dialect='excel', delimiter='\t')
            wr.writerows(data)
    save_tsv(score_dir, act_wps)
    save_tsv(tr_dir, tr)
    save_tsv(te_dir, te)

