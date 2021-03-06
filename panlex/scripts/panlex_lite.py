import os
import pandas as pd
from sqlalchemy import create_engine


min_score = 6
langs_to_keep = ['eng', 'hun', 'deu', 'ita', 'spa', 'fra']

out_fold = '6_lang'


panlex_lite_path = "/mnt/permanent/Language/Multi/Dic/Proj/EmergVocPanLex/panlex_lite/db.sqlite"
engine = create_engine('sqlite:///{0}'.format(panlex_lite_path))

lv = pd.read_sql_table('lv', engine, index_col="lv")

ex = pd.read_sql_table('ex', engine, index_col='ex')

dnx = pd.read_sql_table('dnx', engine)

lv_red = lv[lv.uid.isin(['{0}-000'.format(lang) for lang in langs_to_keep])]
langids = lv_red.index.values

dnx_by_lang = {}
ex_by_lang = {}

for lang in langids:
    print('processing {0}'.format(lang))
    if lang not in ex_by_lang:
        ex_by_lang[lang] = ex[ex.lv==lang]
    if lang not in dnx_by_lang:
        dnx_by_lang[lang] = dnx.merge(ex_by_lang[lang], left_on='ex', right_index=True)

print('will process these langs: {0}'.format([lv.loc[lang]['uid'][:3] for lang in langids]))
done = set()
for lang1 in langids:
    for lang2 in langids:
        lang_pair = tuple(sorted([lang1, lang2]))
        if lang1 == lang2 or lang_pair in done:
            continue
        done.add(lang_pair)
        lang1_name = lv.loc[lang1]['uid'][:3]
        lang2_name = lv.loc[lang2]['uid'][:3]
        print('doing {0}-{1}...'.format(lang1_name, lang2_name))
        ex1 = ex_by_lang[lang1]
        ex2 = ex_by_lang[lang2]
        dnx1 = dnx_by_lang[lang1]
        dnx2 = dnx_by_lang[lang2]

        tr = dnx1.merge(dnx2, on='mn')

        filtered = tr[tr.uq_x + tr.uq_y > min_score]
        print('all: ', len(tr))
        print('len filtered: ', len(filtered))
        filtered = filtered.drop_duplicates(subset=['tt_x', 'tt_y'])
        print('len unique: ', len(filtered))

        filtered_sorted = filtered.sort_values('uq_x', ascending=False)
        print('writing {0}-{1}...'.format(lang1_name, lang2_name))
        fn = os.path.join(out_fold, '{0}_{1}.tsv'.format(lang1_name, lang2_name))

        with open(fn, 'w', encoding='utf-8') as f:
            for _, row in filtered_sorted.iterrows():
                score = (row['uq_x'] + row['uq_y']) / 2
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                    lang1_name, lang2_name, row['tt_x'], row['tt_y'], score))