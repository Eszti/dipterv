from __future__ import print_function
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load_embedding(swadesh_file, embed_file):
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
                i+=1
                continue
            fields = line.strip().decode('utf-8').split(' ')
            w = fields[0]
            if w.lower() in ls_swad:
                trans = fields[1:]
                words.append(w)
                embedding_raw.append(trans)
                if i == len(ls_swad):
                    break
                i+=1 # Reorder embedding
    idx_arr = [words.index(w) for w in ls_swad]
    words_ordered = np.array(words)[idx_arr]
    embedding_ordered = np.array(embedding_raw)[idx_arr]

    # Normalize embedding
    embedding = normalize(embedding_ordered.astype(np.float32))

    return ls_swad, embedding

def get_corrs(embedding, swadesh):
    cnt = embedding.shape[0]
    corr_mx = np.ndarray(shape=(cnt, cnt), dtype=np.float32)

    for i in range(0, cnt):
        for j in range(0, i + 1):
            sim = cosine_similarity(embedding[i].reshape(1, -1), embedding[j].reshape(1, -1))
            corr_mx[i][j] = sim
            corr_mx[j][i] = sim
    sim_mx_args = np.argsort(-corr_mx)
    sims = {}
    for i, w in enumerate(swadesh):
        sims[w] = [swadesh[j] for j in sim_mx_args[i, :]]
    return corr_mx, sim_mx_args, sims

def test():
    en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/eng-000.txt'
    en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'
    en_swad, en_emb = load_embedding(en_swad_fn, en_embed_fn)

    de_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/deu.txt'
    de_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'
    de_swad, de_emb = load_embedding(de_swad_fn, de_embed_fn)


def main():
    test()

if __name__ == '__main__':
    main()