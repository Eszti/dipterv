import json
import numpy as np
from sklearn.preprocessing import normalize

def get_embedding(swadesh_file, swad_idx, embed_file):
    # Read swadesh list
    ls_swad = []
    ls_swad_full = []
    n_found_i = []
    with open(swadesh_file) as f:
        ls_swad = []
        lines = f.read().decode('utf-8').splitlines()
        for (i, line) in enumerate(lines):
            if i not in swad_idx:
                found = False
                if line != '':
                    words = line.split('\t')
                    for word in words:
                        if ' ' not in word:
                            ls_swad.append(word.lower())
                            ls_swad_full.append(word.lower())
                            found = True
                            break
            if not found:
                n_found_i.append(i)
                ls_swad_full.append('NOT_FOUND')

    print('Not found list len: {0}'.format(len(n_found_i)))
    print(ls_swad)
    print(len(ls_swad))

    # Read embeddings
    words = []
    embedding_raw = []
    embed_found_i = []
    with open(embed_file) as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            fields = line.strip().decode('utf-8').split(' ')
            w = fields[0]
            w = w.lower()
            if w in ls_swad:
                embed_found_i.append(ls_swad_full.index(w))
                trans = fields[1:]
                words.append(w)
                embedding_raw.append(trans)
                if i == len(ls_swad):
                    break
                i += 1

    # Delete not found embeddings from swadesh
    # 1. calc not found indices
    # 2. update not found index list
    # 3. update swadesh list
    n_found_i = np.sort(list(set(range(len(ls_swad_full))) - set(embed_found_i)))
    ls_swad = np.delete(ls_swad_full, n_found_i)

    print('Embeddings len: {0}'.format(len(embedding_raw)))
    print('Not found: {0}\n{1}'.format(len(n_found_i), n_found_i))

    # Reorder embedding
    idx_arr = [words.index(w) for w in ls_swad]
    words_ordered = np.array(words)[idx_arr]
    embedding_ordered = np.array(embedding_raw)[idx_arr]

    # Normalize embedding
    embedding = normalize(embedding_ordered.astype(np.float32))

    return ls_swad, embedding, n_found_i

num = 110
silcodes_fn = '/home/eszti/projects/dipterv/univ_embedding/res/swad_fb_{}.json'.format(num)

with open(silcodes_fn) as f:
    silcodes = json.load(f)

sil2fbcodes_fn = '/home/eszti/projects/dipterv/univ_embedding/res/sil2fbcodes.json'
with open(sil2fbcodes_fn) as f:
    sil2fb = json.load(f)

swad_idx = []
en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh{}/deu-000.txt'.format(num)
en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'
en_swad, en_emb, en_nfi = get_embedding(en_swad_fn, swad_idx, en_embed_fn)