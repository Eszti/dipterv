from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def train(W, learning_rate=0.01, num_steps=1001, t1_identity=True):
    num_of_langs = W.shape[0]
    num_of_words = W[0].shape[0]
    dim_of_emb = W[0].shape[1]

    # Init graphs
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_W = tf.constant(W)
        if t1_identity:
            tf_T1 = tf.constant(np.identity(dim_of_emb).astype(np.float32))  # T1 = identity

        # Variables.
        if not t1_identity:
            tf_T1 = tf.Variable(tf.truncated_normal([dim_of_emb, dim_of_emb]))
        tf_T = tf.Variable(tf.truncated_normal([num_of_langs - 1, dim_of_emb, dim_of_emb]))
        tf_A = tf.Variable(tf.truncated_normal([num_of_words, dim_of_emb]))

        # Training computation
        loss = tf.norm(tf.matmul(tf_W[0], tf_T1) - tf_A)  # T1 may be constant
        for i in range(1, num_of_langs):
            loss += tf.norm(tf.matmul(tf_W[i], tf_T[i - 1]) - tf_A)

            # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Run training
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations
            _, l, T1, T, A = session.run([optimizer, loss, tf_T1, tf_T, tf_A])
            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))

        # Print transformation matrices + universal embedding
        print('\n')
        print('Transform 1:')
        print(T1)
        for i in range(0, T.shape[0]):
            print('Transform {}:'.format(i + 2))
            print(T[i])
        print('Universal embedding:')
        print(A)

    # Print transformed embeddings
    print('\n')
    print('W1*T1:')
    print(np.dot(W[0], T1))
    for i in range(0, T.shape[0]):
        print('W{0}*T{0}:'.format(i + 2))
        print(np.dot(W[i + 1], T[i]))

    return (T1, T, A)


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
    print('Valid swadesh len: {0}'.format(len(ls_swad)))

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


def get_corr(embedding, swadesh):
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
    swad_idx = []
    en_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/eng-000.txt'
    en_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.en/wiki.en.vec'
    en_swad, en_emb, en_nfi = get_embedding(en_swad_fn, swad_idx, en_embed_fn)

    de_swad_fn = '/home/eszti/data/panlex_swadesh/swadesh110/test/deu.txt'
    de_embed_fn = '/mnt/permanent/Language/Multi/FB/wiki.de/wiki.de.vec'
    de_swad, de_emb, de_nfi = get_embedding(de_swad_fn, swad_idx, de_embed_fn)

    W = np.ndarray(shape=(2, len(en_swad), en_emb.shape[1]), dtype=np.float32)
    W[0, :, :] = en_emb
    W[1, :, :] = de_emb
    T1, T, A = train(W, num_steps=50000)

    corr_mx, sim_corr, sims_univ = get_corr(A, en_swad)
    print(sims_univ['dog'])

def main():
    test()

if __name__ == '__main__':
    main()