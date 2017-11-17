import json
import logging
import subprocess

import os
import pandas as pd

from embedding import FasttextEmbedding

sil2fb_fn = '/home/eszti/projects/dipterv/notebooks/panlex/data/sil2fb.json'

with open(sil2fb_fn) as f:
     sil2fb = json.load(f)

langs = ['spa', 'fra']
# langs = ['eng', 'hun', 'deu', 'ita', 'spa', 'fra']

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
        df_langs_plx[lang1] = df[[lang1]].drop_duplicates()
    else:
        df_langs_plx[lang1] = pd.concat([df_langs_plx[lang1], df[[lang1]]]).drop_duplicates()
    if lang2 not in df_langs_plx:
        df_langs_plx[lang2] = df[[lang2]].drop_duplicates()
    else:
        df_langs_plx[lang2] = pd.concat([df_langs_plx[lang2], df[[lang2]]]).drop_duplicates()
logging.info('Tables for languages are created!')


# Add found column
for lang in df_langs_plx.keys():
    df_langs_plx[lang] = df_langs_plx[lang].dropna()
    df_langs_plx[lang] = df_langs_plx[lang].assign(found = False)

# Language statistics
header = ['lang', 'words', 'found']
df_lang_stat = pd.DataFrame(columns = header)


# Search for embeddings
for sil in langs:
    fb = sil2fb[sil]

    # Get embedding
    ext = 'bin'
    emb_fn = os.path.join(emb_fold, 'wiki.{0}/wiki.{0}.{1}'.format(fb, ext))
    if ext == 'vec':
        p = subprocess.Popen(['wc', '-l', emb_fn], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        lines = int(result.strip().split()[0])
        logging.info('# lines in embed file: {}'.format(lines))
    # Load embedding
    emb = FasttextEmbedding(emb_fn)

    # Checking if vector in embedding exists for the words
    for i, row in df_langs_plx[sil].iterrows():
        w = row[sil]
        if w in emb.model:
            df_langs_plx[sil].loc[df_langs_plx[sil][sil] == w, 'found'] = True

    # Save and stat
    tot = len(df_langs_plx[sil].index)
    f = len(df_langs_plx[sil][df_langs_plx[sil]['found'] == True])
    logging.info('# words: {}\t# found: {}'.format(tot, f))
    found_fn = os.path.join(out_fold, '{}_found.tsv'.format(sil))
    df_langs_plx[sil].to_csv(found_fn, sep='\t', index=False)
    stat_row = pd.DataFrame([[lang, tot, f]], columns=header)
    df_lang_stat = df_lang_stat.append(stat_row)


# Merge tables
df_merged_dicts = dict()
for lang_pair, df in df_dicts.items():
    lang1 = lang_pair[0]
    lang2 = lang_pair[1]
    tmp1 = pd.merge(df, df_langs_plx[lang1], on=lang1)
    tmp2 = pd.merge(tmp1, df_langs_plx[lang2], on=lang2)
    df_merged_dicts[lang_pair] = tmp2


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