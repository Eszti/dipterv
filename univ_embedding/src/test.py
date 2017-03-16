# coding: utf-8

# # Find universal embedding

# In[1]:

from __future__ import print_function
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# In[12]:

def get_embedding(swadesh_file, embed_file):
    # Read swadesh list
    ls_swad = []
    with open(swadesh_file) as f:
        ls_swad = f.read().decode('utf-8').splitlines()
    ls_swad = [w.lower() for w in ls_swad]

    # Read embeddings
    words = []
    embedding_raw = []
    with open(embed_file) as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            fields = line.strip().decode('utf-8').split(' ')
            w = fields[0]
            if w.lower() in ls_swad:
                trans = fields[1:]
                words.append(w)
                embedding_raw.append(trans)
                if i == len(ls_swad):
                    break
                i += 1

    # Reorder embedding
    idx_arr = [words.index(w) for w in ls_swad]
    words_ordered = np.array(words)[idx_arr]
    embedding_ordered = np.array(embedding_raw)[idx_arr]

    # Normalize embedding
    embedding = normalize(embedding_ordered.astype(np.float32))

    return ls_swad, embedding


# Get Eglish embeddings

# In[14]:

en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/eng-000.txt'
en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'

en_swad, en_emb = get_embedding(en_swad_fn, en_embed_fn)
# print(swad)
# print(emb)


# Get German embeddings

# In[16]:

de_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/deu.txt'
de_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'

de_swad, de_emb = get_embedding(de_swad_fn, de_embed_fn)

# In[18]:

print(de_emb)