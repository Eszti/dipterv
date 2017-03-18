from __future__ import print_function
import numpy as np
import tensorflow as tf

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


def test():
    W1 = np.array([[1, 0], [0, 1], [1, 1]]).astype(np.float32)
    W2 = np.array([[2, 2], [1, -1], [-1, -1]]).astype(np.float32)
    W = np.ndarray(shape=(2, 3, 2), dtype=np.float32)
    W[0, :, :] = W1
    W[1, :, :] = W2
    T1, T, A = train(W)

def main():
    test()

if __name__ == '__main__':
    main()