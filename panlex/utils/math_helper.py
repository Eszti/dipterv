import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate precision
# model_src : source language embeddings (need to have syn0 and index2word properites) (after translation)
# model_tar : target language embeddings (need to have syn0 and index2word properites) (can be don in orig or universal space)
# dict_scr_2_tar : dictionary from source to target
def calc_precision(precs, model_src, model_tar, dict_scr_2_tar, logger):
    W_src = model_src.syn0
    W_tar = model_tar.syn0
    i2w_src = model_src.index2word
    i2w_tar = model_tar.index2word

    return calc_precision_calc(W_src=W_src, i2w_src=i2w_src,
                               W_tar=W_tar, i2w_tar=i2w_tar,
                               precs=precs, dict_scr_2_tar=dict_scr_2_tar,
                               logger=logger)


def calc_precision_calc(W_src, i2w_src, W_tar, i2w_tar, precs, dict_scr_2_tar, logger):
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
    logger.info('sim_mx.shape[0]: {}'.format(sim_mx.shape[0]))
    prec_pcnts = []
    for i, val in enumerate(precs):
        sum_hit = np.sum(prec_cnt[0][0:val])
        pcnt = float(sum_hit) / sim_mx.shape[0]
        logger.debug('prec {} : {}'.format(val, pcnt))
        prec_pcnts.append(pcnt)
    return prec_pcnts

