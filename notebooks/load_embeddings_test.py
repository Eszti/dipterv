
# coding: utf-8

# # Load embeddings

# Load embeddings from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

# In[25]:

from __future__ import print_function
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# In[ ]:

swadesh_codes = ['eng', 'deu', 'spa']
embedding_codes = ['en', 'de', 'es']


# In[13]:

testfile_path = '/mnt/permanent/Language/Multi/FB/wiki.hu/wiki.hu.vec'

words = []
embedding = []

with open(testfile_path) as f:
    i = 0
    for line in f:
        if i == 0:
            i+=1
            continue
        fields = line.strip().decode('utf-8').split(' ')
        w = fields[0]
        trans = fields[1:]
        words.append(w)
        embedding.append(trans)
        i+=1
            
words = np.array(words)
embedding = np.array(embedding)
print(words.shape)
print(embedding.shape)


# Read English swadesh list

# In[170]:

# en_swad_filename = '/home/eszti/data/panlex_swadesh/swadesh110/test/eng-000.txt'
en_swad_filename = '/home/eszti/data/panlex_swadesh/swadesh110/test/deu.txt'


en_swad = []

with open(en_swad_filename) as f:
    en_swad = f.read().decode('utf-8').splitlines()
    
en_swad = [w.lower() for w in en_swad]
en_swad


# In[179]:

# en_embed_filename = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'
en_embed_filename = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'


en_words = []
en_embedding = []

with open(en_embed_filename) as f:
    i = 0
    for line in f:
        if i == 0:
            i+=1
            continue
        fields = line.strip().decode('utf-8').split(' ')
        w = fields[0]
        if w in en_swad:
            trans = fields[1:]
            en_words.append(w)
            en_embedding.append(trans)
            if i == len(en_swad):
                break
            i+=1
          
en_embedding = np.array(en_embedding)
print(len(en_words))
print(en_embedding.shape)


# In[180]:

idx_arr = [en_words.index(w) for w in en_swad]
# idx_arr


# In[184]:

en_words_arr = np.array(en_words)
en_words_ordered = en_words_arr[idx_arr]
en_embedding_ordered = en_embedding[idx_arr]
# print(en_words_ordered)
# print(en_embedding_ordered)


# In[146]:

emb_en = normalize(en_embedding_ordered.astype(np.float32))
# [np.linalg.norm(e) for e in emb_en]
emb_en


# In[181]:

set(en_swad) - set(en_words)


# In[185]:

en_words_ordered


# In[176]:

len(en_swad)


# In[ ]:



