import embeddings
import logging
import sys, time, os
import numpy as np
import json

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s '
                           '%(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

def translate():
    num = 110
    silcodes_fn = '/home/eszti/projects/dipterv/univ_embedding/res/swad_fb_{}.json'.format(num)

    with open(silcodes_fn) as f:
        silcodes = json.load(f)

    sil2fbcodes_fn = '/home/eszti/projects/dipterv/univ_embedding/res/sil2fbcodes.json'
    with open(sil2fbcodes_fn) as f:
        sil2fb = json.load(f)

    swad_idx = []
    en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh{}/eng-000.txt'.format(num)
    en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'
    en_swad, en_emb, en_nfi = embeddings.get_embedding(en_swad_fn, swad_idx, en_embed_fn)

    main_folder = '/home/eszti/data/embeddings/fb_trans/'
    time_str = time.strftime("%H%M")
    date_str = time.strftime("%Y%m%d")
    trans_dir = os.path.join(main_folder, 'trans', '{0}_{1}'.format(date_str, time_str))
    embed_dir = os.path.join(main_folder, 'embedding', '{0}_{1}'.format(date_str, time_str))

    os.makedirs(trans_dir)
    os.makedirs(embed_dir)

    logging.info('making directory for translation matrices: {}'.format(trans_dir))
    logging.info('making directory for embeddings: {}'.format(embed_dir))

    for sil in silcodes:
        if sil == 'eng':
            continue
        logging.info('Translating {} language...'.format(sil))
        swad_fn = '/home/eszti/data/panlex_swadesh/swadesh{0}/{1}-000.txt'.format(num, sil)
        embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.{0}/wiki.{0}.vec'.format(sil2fb[sil])

        logging.info('swadesh file: {}'.format(swad_fn))
        logging.info('embedding file: {}'.format(embed_fn))

        swad, emb, nfi = embeddings.get_embedding(swad_fn, swad_idx, embed_fn)

        missing_words = [w for (i, w) in enumerate(en_swad) if i in nfi]
        logging.info('Missing words: {}'.format(missing_words))

        # Filtered English Swadesh
        en_swad_fil = [w for (i, w) in enumerate(en_swad) if i not in nfi]
        en_emb_fil = np.delete(en_emb, nfi, 0)

        W = np.ndarray(shape=(2, len(swad), emb.shape[1]), dtype=np.float32)
        W[0, :, :] = en_emb_fil
        W[1, :, :] = emb
        T1, T, A = embeddings.train(W, num_steps=50000)

        # Save translation matrix
        trans_fn = os.path.join(trans_dir, 'eng_{}.npy'.format(sil))
        with open(trans_fn, 'w') as f:
            np.save(f, T[0])

        # Calculate missing embeddings
        en_emb_mis = np.take(en_emb, nfi, 0)
        emb_mis = np.dot(en_emb_mis, T[0])

        # Modify embedding
        idx_before = nfi - range(len(nfi))
        mod_embed = np.insert(emb, idx_before, emb_mis, 0)

        # Save modified embedding
        mod_embed_fn = os.path.join(embed_dir, 'eng_{}.npy'.format(sil))
        with open(mod_embed_fn, 'w') as f:
            np.save(f, mod_embed)

def main():
    translate()

if __name__ == '__main__':
    main()