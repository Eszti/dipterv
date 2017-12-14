import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import copy
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import json
from utils import print_verbose, print_debug, save

langs = ['eng', 'ita']
dim = 300
outfolder = 'output'

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
    print_verbose('{0}-{1}\t# word pairs:\t{2}'.format(l1, l1, len(d)))

# Logging training models' length
print_verbose("Traing models' length")
for ((l1, l2), d) in d_wps.items():
    print_verbose('{0}-{1}\t# word pairs:\t{2}'.format(l1, l1, len(d)))

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
            print_debug('{}"\t{}\t{}'.format(key_word, value_words, closest_words))
    if verbose:
        print_debug(prec_cnt)
    prec_pcnts = []
    for i, val in enumerate(precs):
        sum_hit = np.sum(prec_cnt[0][0:val])
        pcnt = float(sum_hit) / sim_mx.shape[0]
        if verbose:
            print_debug('prec {} : {}'.format(val, pcnt))
        prec_pcnts.append(pcnt)
    return prec_pcnts

# Function for training
# langs : list containing sil codes of languages, tf_T follows the same order
# d_models : dictionary, lang - embedding (KeyedVectors)
# d_wps : dictiorary, lang_pair - wordpair list
# d_tr_mods: dictionary, lang_pair - embedding (KeyedVectors) only words used for training
# dim : dimensiion of embedding
# epochs : epochs to run
# precs_to_calc : list of precisions to calculate, e.g. [1,3,5] if we want Precision @1, @3, @5
# iters : optional, break after n update
# lr : learning rate
# svd : whether to do svd regularization
# svd_f : how often regularize
# verbose : print out details
def train(langs, d_models, d_wps, d_tr_mods, dim, epochs, precs_to_calc, iters=None, lr=0.3, svd=False, svd_f=1,
          verbose=False):
    nb_langs = len(langs)

    # Init graphs
    graph = tf.Graph()
    with graph.as_default():
        # TF variables
        # Placeholder for 2 words
        tf_w1 = tf.placeholder(tf.float32, shape=[None, dim])
        tf_w2 = tf.placeholder(tf.float32, shape=[None, dim])
        # Placeholder for indexing the T matrix
        tf_idx_l1 = tf.placeholder(tf.int32)
        tf_idx_l2 = tf.placeholder(tf.int32)
        # Translation matrices
        tf_T = tf.Variable(tf.truncated_normal([nb_langs, dim, dim]))

        # SVD reguralization
        tf_s1, tf_U1, tf_V1 = tf.svd(tf_T[tf_idx_l1], full_matrices=True, compute_uv=True)
        updated_1 = tf.assign(tf_T[tf_idx_l1], tf.matmul(tf_U1, tf_V1))
        tf_s2, tf_U2, tf_V2 = tf.svd(tf_T[tf_idx_l2], full_matrices=True, compute_uv=True)
        updated_2 = tf.assign(tf_T[tf_idx_l2], tf.matmul(tf_U2, tf_V2))

        # Loss
        tf_T1 = tf.matmul(tf_w1, tf_T[tf_idx_l1])
        tf_T2 = tf.matmul(tf_w2, tf_T[tf_idx_l2])
        tf_T1_n = tf.nn.l2_normalize(tf_T1, dim=1)
        tf_T2_n = tf.nn.l2_normalize(tf_T2, dim=1)
        loss = tf.matmul(tf_T1_n, tf.transpose(tf_T2_n))
        loss = -loss

        # Applying optimizer, Todo: try different optimizers!!
        # https://www.tensorflow.org/api_guides/python/train#Optimizers
        optimizer = tf.train.AdagradOptimizer(lr).minimize(loss)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        j = 0
        lc_arr = []
        precs_arr = []
        for i in range(epochs):
            loss_arr = []
            for ((l1, l2), wp_l) in d_wps.items():
                loss_arr_l = []
                idx_l1 = langs.index(l1)
                idx_l2 = langs.index(l2)
                k = 0
                for (w1, w2) in wp_l:
                    emb1 = d_models[l1][w1].reshape((1, 300))
                    emb2 = d_models[l2][w2].reshape((1, 300))
                    # Todo: if we add "or j == 0" for some reason it's better in this mock example
                    if (svd and i % svd_f == 0) or j == 0:
                        _, l, _, _, T = session.run([optimizer, loss, updated_1, updated_2, tf_T],
                                                    feed_dict={tf_w1: emb1,
                                                               tf_w2: emb2,
                                                               tf_idx_l1: idx_l1,
                                                               tf_idx_l2: idx_l2})
                    else:
                        _, l, T = session.run([optimizer, loss, tf_T],
                                              feed_dict={tf_w1: emb1,
                                                         tf_w2: emb2,
                                                         tf_idx_l1: idx_l1,
                                                         tf_idx_l2: idx_l2})
                    j += 1
                    k += 1
                    loss_arr.append(-l[0][0])
                    loss_arr_l.append(-l[0][0])
                    if iters is not None and j == iters:
                        break
                loss_np_arr_l = np.asarray(loss_arr_l)
                if j % 100 == 0:
                    print_verbose('iter: {3}\t{0} - {1}\tavg loss: {2}'.format(l1, l2, np.average(loss_np_arr_l), j))
                    save(T=T, lc=lc, precs=precs, i=j)
                if iters is not None and j == iters:
                    break

            # Monitoring for learning curve
            loss_np_arr = np.asarray(loss_arr)
            loss_epoch_avg = np.average(loss_np_arr)
            print_verbose('{0}\tavg sims: {1}'.format(i, loss_epoch_avg))
            lc_arr.append(loss_epoch_avg)

            # Calculate precision
            e_prec_l = []
            for ((l1, l2), _) in d_wps.items():
                m1_tr = copy.deepcopy(d_tr_mods[l1])
                m2_tr = copy.deepcopy(d_tr_mods[l2])
                m1 = copy.deepcopy(d_models[l1])
                m2 = copy.deepcopy(d_models[l2])
                # Get translations matrices
                idx_l1 = langs.index(l1)
                idx_l2 = langs.index(l2)
                T1 = T[idx_l1]
                T2 = T[idx_l2]
                m1.syn0 = np.dot(m1.syn0, T1)
                m2.syn0 = np.dot(m2.syn0, T2)
                precs_1 = calc_precision(precs_to_calc, m1, m2, d_dict[(l1, l2)], verbose=False)
                precs_2 = calc_precision(precs_to_calc, m2, m1, d_dict[(l2, l1)], verbose=False)
                # Todo: should be done this way
                # # Prec l1 - l2
                # m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
                # precs_1 = calc_precision(precs_to_calc, m1_tr, m2, d_dict[(l1, l2)], verbose=False)
                # # Prec l2 - l1
                # m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
                # precs_2 = calc_precision(precs_to_calc, m2_tr, m1, d_dict[(l2, l1)], verbose=False)
                e_prec_l.append(((l1, l2), precs_1))
                e_prec_l.append(((l2, l1), precs_2))
            print_verbose(e_prec_l)
            precs_arr.append(e_prec_l)
    return T, lc_arr, precs_arr


T, lc, precs = train(langs, d_models, d_wps, d_tr_mods, 300, 20, [1, 3, 5])
save(T=T, lc=lc, precs=precs)

