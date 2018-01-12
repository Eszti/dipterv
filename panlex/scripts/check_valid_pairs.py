import json
import logging
import subprocess

import os
import pandas as pd

from OLD_TO_DEL.embedding import Word2VecEmbedding

sil2fb_fn = '/home/eszti/projects/dipterv/notebooks/panlex/data/sil2fb.json'

with open(sil2fb_fn) as f:
     sil2fb = json.load(f)

# langs = ['spa', 'fra']
langs = ['eng', 'hun', 'deu', 'ita', 'spa', 'fra']

pan_fold = '6_lang'
emb_fold = '/mnt/permanent/Language/Multi/FB'
out_fold = '6_lang_out'
loglevel = logging.INFO

logging.basicConfig(filename=os.path.join(out_fold, 'log.txt'),
                    level=loglevel,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filemode='w')

# Read all panlex dictionaries
df_dicts = dict()
logging.info('will process these langs: {0}'.format(langs))
done = set()
for lang1 in langs:
    for lang2 in langs:
        lang_pair = tuple(sorted([lang1, lang2]))
        if lang1 == lang2 or lang_pair in done:
            continue
        done.add(lang_pair)
        pan_fn = os.path.join(pan_fold, '{0}_{1}.tsv'.format(lang_pair[0], lang_pair[1]))
        logging.info('Reading dictionary from: {0}'.format(pan_fn))
        df = pd.read_csv(pan_fn, sep='\t', header=None, names = ['lang 1', "lang 2", lang_pair[0],  lang_pair[1], 'score'])
        df_dicts[lang_pair] = df


# Create tables for each language
df_langs_plx = dict()
for lang_pair, df in df_dicts.items():
    lang1 = lang_pair[0]
    lang2 = lang_pair[1]
    if lang1 not in df_langs_plx:
        df_langs_plx[lang1] = df[[lang1]].drop_duplicates().dropna()
    else:
        df_langs_plx[lang1] = pd.concat([df_langs_plx[lang1], df[[lang1]]]).drop_duplicates().dropna()
    if lang2 not in df_langs_plx:
        df_langs_plx[lang2] = df[[lang2]].drop_duplicates().dropna()
    else:
        df_langs_plx[lang2] = pd.concat([df_langs_plx[lang2], df[[lang2]]]).drop_duplicates().dropna()
logging.info('Tables for languages are created!')


# Language statistics
header = ['lang', 'words', 'found', 'emb']
df_lang_stat = pd.DataFrame(columns = header)


# Search for embeddings
df_langs_found = dict()
for sil in langs:
    fb = sil2fb[sil]

    # Get embedding
    ext = 'vec'
    emb_fn = os.path.join(emb_fold, 'wiki.{0}/wiki.{0}.{1}'.format(fb, ext))
    if ext == 'vec':
        p = subprocess.Popen(['wc', '-l', emb_fn], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        lines = int(result.strip().split()[0])
        logging.info('# lines in embed file: {}'.format(lines))
    # Load embedding
    emb = Word2VecEmbedding(emb_fn, 'word2vec_txt')

    # Checking if embedding can be found for panlex vocab words
    n_pan = len(df_langs_plx[sil].index)
    logging.info('# panlex vocab:  {}'.format(n_pan))
    vocab = set(emb.model.vocab.keys())
    df_emb = pd.DataFrame(list(vocab), columns=[sil])
    n_emb = len(df_emb.index)
    logging.info('# emb vocab:  {}'.format(n_emb))
    df_found = df_langs_plx[sil].merge(df_emb)
    df_found = df_found.assign(found = True)
    df_langs_found[sil] = df_found
    n_fou = len(df_found.index)
    logging.info('# found vocab:  {}'.format(n_fou))

    # Stat
    stat_row = pd.DataFrame([[sil, n_pan, n_fou, n_emb]], columns=header)
    df_lang_stat = df_lang_stat.append(stat_row)


# Merge tables
df_merged_dicts = dict()
for lang_pair, df in df_dicts.items():
    lang1 = lang_pair[0]
    lang2 = lang_pair[1]
    tmp1 = pd.merge(df, df_langs_found[lang1], on=lang1, how='left')
    tmp2 = pd.merge(tmp1, df_langs_found[lang2], on=lang2, how='left')
    df_merged_dicts[lang_pair] = tmp2.fillna(False)
    merge_fn = os.path.join(out_fold, '{0}_{1}.tsv'.format(lang1, lang2))
    df_merged_dicts[lang_pair].to_csv(merge_fn, sep='\t', index=False)


# Save language statistics
fn = os.path.join(out_fold, 'df_lang_stat.tsv')
df_lang_stat.to_csv(fn, sep='\t', index=False)


# Panlex statistics
header = ['lang1', 'lang2', 'word_pairs', 'found']
df_plx_stat = pd.DataFrame(columns = header)
for lang_pair, df in df_merged_dicts.items():
    lang1 = lang_pair[0]
    lang2 = lang_pair[1]
    l = len(df.index)
    f = len(df[(df['found_x'] == True) & (df['found_y'] == True)])
    row = pd.DataFrame([[lang1, lang2, l, f]], columns = header)
    df_plx_stat = df_plx_stat.append(row)

fn = os.path.join(out_fold, 'df_plx_stat.tsv')
df_plx_stat.to_csv(fn, sep='\t', index=False)