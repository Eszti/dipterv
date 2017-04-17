import utils
import logging
import sys, os, time
import numpy as np
import json
from sklearn.preprocessing import normalize

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s '
                           '%(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

def translate(cfg, starttime):
    # Config values
    cfg_trans = utils.get_translate_config(cfg)
    # Logging
    # General
    logging.info('Output folder: {}'.format(cfg_trans.output_root))
    # Codes
    logging.info('Silcodes file: {}'.format(cfg_trans.silcodes_fn))
    logging.info('Silcodes mapping: {}'.format(cfg_trans.sil2fb_map))
    # Embedding
    logging.info('Embedding dir: {}'.format(cfg_trans.emb_dir))
    logging.info('English embedding file: {}'.format(cfg_trans.eng_emb_fn))
    # Swadesh
    logging.info('Swadesh dir: {}'.format(cfg_trans.swad_dir))
    # Train
    logging.info('Verbose: {}'.format(cfg_trans.verbose))
    logging.info('Logging frequency: {}'.format(cfg_trans.log_freq))
    logging.info('Number of steps: {}'.format(cfg_trans.num_steps))
    logging.info('End condition: {}'.format(cfg_trans.end_cond))
    logging.info('Learning rate: {}'.format(cfg_trans.learning_rate))

    with open(cfg_trans.silcodes_fn) as f:
        silcodes = json.load(f)

    logging.info('Number of languages: {}'.format(len(silcodes)))

    with open(cfg_trans.sil2fb_map) as f:
        sil2fb = json.load(f)

    swad_idx = []
    en_swad, en_emb, en_nfi = utils.get_embedding(cfg_trans.eng_swad_fn, swad_idx, cfg_trans.eng_emb_fn)

    trans_dir = os.path.join(cfg_trans.output_root, 'trans')
    embed_dir = os.path.join(cfg_trans.output_root, 'embedding')

    trans_dir = utils.create_timestamped_dir(trans_dir)
    embed_dir = utils.create_timestamped_dir(embed_dir)

    logging.info('making directory for translation matrices: {}'.format(trans_dir))
    logging.info('making directory for embeddings: {}'.format(embed_dir))

    for sil in silcodes:
        if sil == 'eng':
            continue
        logging.info('Translating {} language...'.format(sil.upper()))
        swad_fn = os.path.join(cfg_trans.swad_dir, '{}-000.txt'.format(sil))
        embed_fn = os.path.join(cfg_trans.emb_dir, 'wiki.{0}/wiki.{0}.vec'.format(sil2fb[sil]))

        logging.info('swadesh file: {}'.format(swad_fn))
        logging.info('embedding file: {}'.format(embed_fn))

        try:
            swad, emb, nfi = utils.get_embedding(swad_fn, swad_idx, embed_fn)
        except:
            logging.warning('Skipping language {0}'.format(sil))
            continue

        missing_words = [w for (i, w) in enumerate(en_swad) if i in nfi]
        logging.info('Missing words: {}'.format(missing_words))

        # Filtered English Swadesh
        en_swad_fil = [w for (i, w) in enumerate(en_swad) if i not in nfi]
        en_emb_fil = np.delete(en_emb, nfi, 0)

        W = np.ndarray(shape=(2, len(swad), emb.shape[1]), dtype=np.float32)
        W[0, :, :] = en_emb_fil
        W[1, :, :] = emb
        T1, T, A = utils.train(W, num_steps=cfg_trans.num_steps, learning_rate=cfg_trans.learning_rate,
                               verbose=cfg_trans.verbose, log_freq=cfg_trans.log_freq,
                               end_cond=cfg_trans.end_cond, max_iter=cfg_trans.max_iter, debug=cfg_trans.debug)

        # Save translation matrix
        trans_fn = os.path.join(trans_dir, 'eng_{}.npy'.format(sil))
        logging.info('Saving translation mx to {}'.format(trans_fn))
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
        norm_embed = normalize(mod_embed.astype(np.float32))           # NORMALIZING!!
        logging.info('Saving embedding to {}'.format(mod_embed_fn))
        with open(mod_embed_fn, 'w') as f:
            np.save(f, norm_embed)

def main():
    starttime = int(round(time.time()))
    os.nice(20)
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    logging.info('Config file: {}'.format(cfg_file))
    cfg = utils.get_cfg(cfg_file)
    translate(cfg, starttime)
    finishtime = int(round(time.time()))
    logging.info('Running time in seconds: {}'.format(finishtime - starttime))


if __name__ == '__main__':
    main()