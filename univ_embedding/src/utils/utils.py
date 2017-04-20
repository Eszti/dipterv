import numpy as np

def find_all_indices(list, val):
    idxs = [i for i, elem in enumerate(list) if elem == val]
    return idxs

def get_rowwise_norm(embedding):
    sum = 0
    for row in embedding:
        norm = np.linalg.norm(row)
        sum += norm
    return sum

def save_nparr(embed_fn, emb):
    with open(embed_fn, 'w') as f:
        np.save(f, emb)