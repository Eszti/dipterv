import numpy as np
import train
import embeddings

def main():
    en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/eng-000.txt'
    en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'
    print 'Loading {0}'.format(en_embed_fn)
    en_swad, en_emb = embeddings.load_embedding(en_swad_fn, en_embed_fn)
    print 'Swadesh len: {0}'.format(len(en_swad))
    print 'Embedding shape: {0}'.format(en_emb.shape)

    de_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/deu.txt'
    de_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'
    print 'Loading {0}'.format(de_embed_fn)
    de_swad, de_emb = embeddings.load_embedding(de_swad_fn, de_embed_fn)
    print 'Swadesh len: {0}'.format(len(de_swad))
    print 'Embedding shape: {0}'.format(de_emb.shape)

    W = np.ndarray(shape=(2, len(en_swad), en_emb.shape[1]), dtype=np.float32)
    W[0, :, :] = en_emb
    W[1, :, :] = de_emb
    T1, T, A = train.train(W)

if __name__ == '__main__':
    main()