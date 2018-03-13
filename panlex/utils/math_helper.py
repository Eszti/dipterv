import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate precision
# model_src : source language embeddings (need to have syn0 and index2word properites) (after translation)
# model_tar : target language embeddings (need to have syn0 and index2word properites) (can be don in orig or universal space)
# dict_scr_2_tar : dictionary from source to target
def calc_precision_keyedvec(precs, model_src, model_tar, dict_scr_2_tar, logger):
    W_src = model_src.syn0
    W_tar = model_tar.syn0
    i2w_src = model_src.index2word
    i2w_tar = model_tar.index2word

    return calc_precision(W_src=W_src, i2w_src=i2w_src,
                          W_tar=W_tar, i2w_tar=i2w_tar,
                          precs=precs, dict_scr_2_tar=dict_scr_2_tar,
                          logger=logger)


def calc_precision(W_src, i2w_src, W_tar, i2w_tar, precs, dict_scr_2_tar, logger):
    cos_mx = cosine_similarity(W_src, W_tar)
    sim_mx = np.argsort(-cos_mx, axis=1)
    max_prec = max(precs)
    prec_cnt = np.zeros(shape=(1, max_prec))
    logger.debug('word: \ttranslations in dict: \tclosest words after translation: \t')
    for i, r in enumerate(sim_mx):
        key_word = i2w_src[i]
        value_words = dict_scr_2_tar[key_word]
        closest_words = []
        for j in range(max_prec):
            word = i2w_tar[r[j]]
            closest_words.append(word)
            if word in value_words:
                prec_cnt[0][j] = prec_cnt[0][j] + 1
                break
        logger.debug('{}"\t{}\t{}'.format(key_word, value_words, closest_words))
    logger.debug(prec_cnt)
    prec_pcnts = []
    for i, val in enumerate(precs):
        sum_hit = np.sum(prec_cnt[0][0:val])
        pcnt = float(sum_hit) / sim_mx.shape[0]
        logger.debug('prec {} : {}'.format(val, pcnt))
        prec_pcnts.append(pcnt)
    return prec_pcnts

def calc_loss(M1, M2):
     cos_mx = np.sum(np.multiply(M1, M2), axis=1)
     avg = np.average(cos_mx)
     return avg

def get_indexes_of_wplist(wp_l, emb_l1, emb_l2):
    l1_idxs = []
    l2_idxs = []
    for w1, w2 in wp_l:
        idx_w1 = emb_l1.index2word.index(w1)
        idx_w2 = emb_l2.index2word.index(w2)
        l1_idxs.append(idx_w1)
        l2_idxs.append(idx_w2)
    return l1_idxs, l2_idxs

def get_embeddings_for_batch(emb_dict, wp_l, dim, l1, l2):
    nb_rows = len(wp_l)
    emb1 = np.ndarray(shape=(nb_rows, dim))
    emb2 = np.ndarray(shape=(nb_rows, dim))
    for (i, (w1, w2)) in enumerate(wp_l):
        emb1[i, :] = emb_dict[l1][w1]
        emb2[i, :] = emb_dict[l2][w2]
    return emb1, emb2

def gather(M, idxs):
    return np.take(M, idxs, axis=0)

def prec_at(ranks, cut):
    return len([r for r in ranks if r <= cut])/float(len(ranks))

def get_rank(nn, gold):
    for idx,word in enumerate(nn):
        if word in gold:
            return idx + 1
    return idx + 1

def score(sp1, sp2, gold, additional):
    sim_mat = -sp2.syn0 * sp1.syn0.T

    if additional:
        # for each element, computes its rank in the ranked list of
        # similarites. sorting done on the opposite axis (inverse querying)
        srtd_idx = np.argsort(np.argsort(sim_mat, axis=1), axis=1)

        # for each element, the resulting rank is combined with cosine scores.
        # the effect will be of breaking the ties, because cosines are smaller
        # than 1. sorting done on the standard axis (regular NN querying)
        srtd_idx = np.argsort(srtd_idx + sim_mat, axis=0)
    else:
        srtd_idx = np.argsort(sim_mat, axis=0)

    ranks = []
    for i, el1 in enumerate(gold.keys()):

        sp1_idx = sp1.row2id[el1]

        # print the top 5 translations
        translations = []
        for j in range(5):
            sp2_idx = srtd_idx[j, sp1_idx]
            word, score = sp2.id2row[sp2_idx], -sim_mat[sp2_idx, sp1_idx]
            translations.append("\t\t%s:%.3f" % (word, score))

        translations = "\n".join(translations)

        # get the rank of the (highest-ranked) translation
        rnk = get_rank(srtd_idx[:, sp1_idx].A.ravel(),
                       [sp2.row2id[el] for el in gold[el1]])
        ranks.append(rnk)

        print("\nId: %d Source: %s \n\tTranslation:\n%s \n\tGold: %s \n\tRank: %d" %
              (len(ranks), el1, translations, gold[el1], rnk))

    print('Corrected: %s' % str(additional))

    if additional:
        print('Total extra elements, Test(%d) + Additional:%d' % (len(gold.keys()),
                                                            sp1.mat.shape[0]))
    for k in [1, 5, 10]:
        print('Prec@%d: %.3f' % (k, prec_at(ranks, k)))