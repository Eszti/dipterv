
# coding: utf-8

# # Functions

# In[1]:

from __future__ import print_function
import numpy as np
import tensorflow as tf


# In[3]:

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
          tf_T1 = tf.constant(np.identity(dim_of_emb).astype(np.float32))    # T1 = identity

      # Variables.
      if not t1_identity:
        tf_T1 = tf.Variable(tf.truncated_normal([dim_of_emb, dim_of_emb]))
      tf_T = tf.Variable(tf.truncated_normal([num_of_langs-1, dim_of_emb, dim_of_emb])) 
      tf_A = tf.Variable(tf.truncated_normal([num_of_words, dim_of_emb]))

      # Training computation
      loss = tf.norm(tf.matmul(tf_W[0], tf_T1) - tf_A)                   # T1 may be constant
      for i in range(1, num_of_langs):
        loss += tf.norm(tf.matmul(tf_W[i], tf_T[i-1]) - tf_A) 

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
          print('Transform {}:'.format(i+2))
          print(T[i])
      print('Universal embedding:')
      print(A)

    # Print transformed embeddings
    print('\n')
    print('W1*T1:')
    print(np.dot(W[0], T1))
    for i in range(0, T.shape[0]):
      print('W{0}*T{0}:'.format(i+2))
      print(np.dot(W[i+1], T[i])) 
        
    return (T1, T, A)


# In[4]:

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
                i+=1
                
    # Reorder embedding
    idx_arr = [words.index(w) for w in ls_swad]
    words_ordered = np.array(words)[idx_arr]
    embedding_ordered = np.array(embedding_raw)[idx_arr]
    
    # Normalize embedding
    embedding = normalize(embedding_ordered.astype(np.float32))
    
    return ls_swad, embedding


# In[ ]:



