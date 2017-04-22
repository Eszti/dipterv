import logging
import time

import numpy as np
import os
import tensorflow as tf


def _log_steps(l, step, starttime):
    currtime = int(round(time.time()))
    logging.info('Loss at step %d: %f (tiem: %d)' % (step, l, currtime - starttime))

def _save_nparr(embed_fn, emb):
    with open(embed_fn, 'w') as f:
        np.save(f, emb)

def save_train_progress(output_dir, T1, T, A, step):
    if output_dir is not None:
        logging.info('Saving T1, T, A into {}'.format(output_dir))
        T1_fn = os.path.join(output_dir, 'T1_{}.npy'.format(step))
        T_fn = os.path.join(output_dir, 'T_{}.npy'.format(step))
        A_fn = os.path.join(output_dir, 'A_{}.npy'.format(step))
        _save_nparr(T1_fn, T1)
        _save_nparr(T_fn, T)
        _save_nparr(A_fn, A)

def train(W, learning_rate=0.01, num_steps=1001, t1_identity=True, loss_crit=0.0001,
          loss_crit_flag = True, output_dir=None, end_cond=None, max_iter=None, verbose=False):
    if output_dir is not None:
        os.makedirs(output_dir)
    starttime = int(round(time.time()))
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
        logging.info('Training session is initialized, starting training...')
        step = 0
        l = float("inf")
        l_prev = l
        while ((step < num_steps) and (num_steps != 0)) or ((num_steps == 0) and (end_cond < l)):
            if (num_steps == 0) and max_iter is not None and step > max_iter:
                break
            # Run the computations
            _, l, T1, T, A = session.run([optimizer, loss, tf_T1, tf_T, tf_A])
            if (step % 10000 == 0):
                _log_steps(l, step, starttime)
            if (step % 100000 == 0) and verbose:
                save_train_progress(output_dir, T1, T, A, step)
            if loss_crit_flag:
                if abs(l - l_prev) < loss_crit:
                    logging.info('Loss does not change anymore ({0}, {1}), finishing training at step: {2}'
                                 .format(l_prev, l, step))
                    break
            l_prev = l
            step += 1
        _log_steps(l, step, starttime)
        save_train_progress(output_dir, T1, T, A, step)
        logging.info('Finishing training...')
    return T1, T, A