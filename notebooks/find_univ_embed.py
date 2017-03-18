
# coding: utf-8

# # Find universal embedding

# In[14]:

from __future__ import print_function
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
def execute_notebook(nbfile):    
    with io.open(nbfile) as f:
        nb = current.read(f, 'json')
    
    ip = get_ipython()
    
    for cell in nb.worksheets[0].cells:
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.input)


# In[15]:

execute_notebook("functions.ipynb")


# Get Eglish embeddings

# In[16]:

en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/eng-000.txt'
en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'

en_swad, en_emb = get_embedding(en_swad_fn, en_embed_fn)
# print(swad)
# print(emb)


# Get German embeddings

# In[17]:

de_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/deu.txt'
de_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'

de_swad, de_emb = get_embedding(de_swad_fn, de_embed_fn)


# In[19]:

_, _, sims_en = get_corr(en_emb, en_swad)
sims_en['dog']


# In[37]:

_, _, sims_de = get_corr(de_emb, de_swad)
sims_de['hund']


# Train

# In[39]:

W = np.ndarray(shape=(2, len(en_swad), en_emb.shape[1]), dtype=np.float32)
W[0, :, :] = en_emb
W[1, :, :] = de_emb
T1, T, A = train(W, num_steps=50000)


# In[42]:

corr_mx, sim_corr, sims_univ = get_corr(A, en_swad)


# In[43]:

corr_mx


# In[44]:

sims_univ['black']

