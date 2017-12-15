from base.loggable import Loggable
import tensorflow as tf

class TrainMModel(Loggable):
    def __init__(self, train_config, data_model_wrapper, language_config):
        Loggable.__init__(self)
        self.train_config = train_config
        self.data_model_wrapper = data_model_wrapper
        self.langs = language_config.langs
        self.dim = data_model_wrapper.dim

    def train(self):
        nb_langs = len(self.langs)

        # Init graphs
        graph = tf.Graph()
        with graph.as_default():
            # TF variables
            # Placeholder for 2 words
            tf_w1 = tf.placeholder(tf.float32, shape=[None, dim])
            tf_w2 = tf.placeholder(tf.float32, shape=[None, dim])
            # Placeholder for indexing the T matrix
            tf_idx_l1 = tf.placeholder(tf.int32)
            tf_idx_l2 = tf.placeholder(tf.int32)
            # Translation matrices
            tf_T = tf.Variable(tf.truncated_normal([nb_langs, dim, dim]))

            # SVD reguralization
            tf_s1, tf_U1, tf_V1 = tf.svd(tf_T[tf_idx_l1], full_matrices=True, compute_uv=True)
            updated_1 = tf.assign(tf_T[tf_idx_l1], tf.matmul(tf_U1, tf_V1))
            tf_s2, tf_U2, tf_V2 = tf.svd(tf_T[tf_idx_l2], full_matrices=True, compute_uv=True)
            updated_2 = tf.assign(tf_T[tf_idx_l2], tf.matmul(tf_U2, tf_V2))

            # Loss
            tf_T1 = tf.matmul(tf_w1, tf_T[tf_idx_l1])
            tf_T2 = tf.matmul(tf_w2, tf_T[tf_idx_l2])
            tf_T1_n = tf.nn.l2_normalize(tf_T1, dim=1)
            tf_T2_n = tf.nn.l2_normalize(tf_T2, dim=1)
            loss = tf.matmul(tf_T1_n, tf.transpose(tf_T2_n))
            loss = -loss

            # Applying optimizer, Todo: try different optimizers!!
            # https://www.tensorflow.org/api_guides/python/train#Optimizers
            optimizer = tf.train.AdagradOptimizer(lr).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            j = 0
            lc_arr = []
            precs_arr = []
            for i in range(epochs):
                loss_arr = []
                for ((l1, l2), wp_l) in d_wps.items():
                    loss_arr_l = []
                    idx_l1 = langs.index(l1)
                    idx_l2 = langs.index(l2)
                    k = 0
                    for (w1, w2) in wp_l:
                        emb1 = d_models[l1][w1].reshape((1, 300))
                        emb2 = d_models[l2][w2].reshape((1, 300))
                        # Todo: if we add "or j == 0" for some reason it's better in this mock example
                        if (svd and i % svd_f == 0) or j == 0:
                            _, l, _, _, T = session.run([optimizer, loss, updated_1, updated_2, tf_T],
                                                        feed_dict={tf_w1: emb1,
                                                                   tf_w2: emb2,
                                                                   tf_idx_l1: idx_l1,
                                                                   tf_idx_l2: idx_l2})
                        else:
                            _, l, T = session.run([optimizer, loss, tf_T],
                                                  feed_dict={tf_w1: emb1,
                                                             tf_w2: emb2,
                                                             tf_idx_l1: idx_l1,
                                                             tf_idx_l2: idx_l2})
                        j += 1
                        k += 1
                        loss_arr.append(-l[0][0])
                        loss_arr_l.append(-l[0][0])
                        if iters is not None and j == iters:
                            break
                    if iters is not None and j == iters:
                        break

                # Monitoring for learning curve
                loss_np_arr = np.asarray(loss_arr)
                loss_epoch_avg = np.average(loss_np_arr)
                print_verbose('epoch:\t{0}\tavg sims: {1}'.format(i, loss_epoch_avg))
                lc_arr.append(loss_epoch_avg)

                # Calculate precision
                e_prec_l = []
                for ((l1, l2), _) in d_wps.items():
                    print_verbose('Calculating precision for {0}-{1}'.format(l1, l2))
                    m1_tr = copy.deepcopy(d_tr_mods[(l1, l2)])
                    m2_tr = copy.deepcopy(d_tr_mods[(l2, l1)])
                    # m1 = copy.deepcopy(d_models[l1])
                    # m2 = copy.deepcopy(d_models[l2])
                    # Get translations matrices
                    idx_l1 = langs.index(l1)
                    idx_l2 = langs.index(l2)
                    T1 = T[idx_l1]
                    T2 = T[idx_l2]
                    m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
                    m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
                    precs_1 = calc_precision(precs_to_calc, m1_tr, m2_tr, d_dict[(l1, l2)], verbose=False)
                    precs_2 = calc_precision(precs_to_calc, m2_tr, m1_tr, d_dict[(l2, l1)], verbose=False)
                    # Todo: should be done this way
                    # # Prec l1 - l2
                    # m1_tr.syn0 = np.dot(m1_tr.syn0, T1)
                    # precs_1 = calc_precision(precs_to_calc, m1_tr, m2, d_dict[(l1, l2)], verbose=False)
                    # # Prec l2 - l1
                    # m2_tr.syn0 = np.dot(m2_tr.syn0, T2)
                    # precs_2 = calc_precision(precs_to_calc, m2_tr, m1, d_dict[(l2, l1)], verbose=False)
                    e_prec_l.append(((l1, l2), precs_1))
                    e_prec_l.append(((l2, l1), precs_2))
                print_verbose(e_prec_l)
                precs_arr.append(e_prec_l)
                save(T=T, i=i)
        return T, lc_arr, precs_arr

    def run(self):
        pass