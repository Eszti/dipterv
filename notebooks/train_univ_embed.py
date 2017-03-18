
# coding: utf-8

# # Universal embeddings: small test

# Finding a universal embedding that resemles the other embeddings the most. 

# In[1]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import colorsys


# In[2]:

global_colors = ['blue', 'gold', 'red']


# Function for training (init + run):

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


# Fuction for printing results:

# In[42]:

def print_results(W, T1, T, A, use_global_colors=True):
    num_of_langs = W.shape[0]
    num_of_words = W[0].shape[0]
    dim_of_emb = W[0].shape[1]
    
    if use_global_colors:
        colors = global_colors
    else:
        HSV_tuples = [(x*1.0/num_of_langs, 0.5, 0.5) for x in range(num_of_langs)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        colors = RGB_tuples

    Xs = np.ndarray(shape=(num_of_langs, num_of_words))
    Ys = np.ndarray(shape=(num_of_langs, num_of_words))
    Xs[0, :] = np.dot(W[0], T1)[:, 0]
    Ys[0, :] = np.dot(W[0], T1)[:, 1]
    for i in range(1, num_of_langs):
        Xs[i, :] = np.dot(W[i], T[i-1])[:, 0]
        Ys[i, :] = np.dot(W[i], T[i-1])[:, 1]

    for i in range(num_of_langs):
        plt.scatter(Xs[i], Ys[i], color=colors[i])
    
    plt.grid()
    plt.show()

    A_xs = A[:, 0]
    A_ys = A[:, 1]
    for i in range(num_of_langs):
        plt.scatter(Xs[i], Ys[i], color=colors[i])
    plt.scatter(A_xs, A_ys, color='black')

    print('Black points = universal embedding')
    plt.grid()
    plt.show()


# #### 2 language - 2 words example  
# 
# Diagram:  
# <span style="color:blue">**W1 points**</span>  
# <span style="color:gold">**W2 points**</span>  

# In[43]:

W1 = np.array([[1, 0], [0, 1]]).astype(np.float32)
W2 = np.array([[1, 1], [1, -1]]).astype(np.float32)

W = np.ndarray(shape=(2,2,2), dtype=np.float32)
W[0, :, :] = W1
W[1, :, :] = W2

W1_xs = W[0][:, 0]
W1_ys = W[0][:, 1]
plt.scatter(W1_xs, W1_ys, color='blue')
W2_xs = W[1][:, 0]
W2_ys = W[1][:, 1]
plt.scatter(W2_xs, W2_ys, color='gold')

plt.grid()
plt.show()


# In[44]:

T1, T, A = train(W)
print_results(W, T1, T, A)


# #### 2 language - 3 words example
# 
# Diagram:  
# <span style="color:blue">**W1 points**</span>  
# <span style="color:gold">**W2 points**</span>  

# In[45]:

W1 = np.array([[1, 0], [0, 1], [1, 1]]).astype(np.float32)
W2 = np.array([[2, 2], [1, -1], [-1, -1]]).astype(np.float32)


W = np.ndarray(shape=(2,3,2), dtype=np.float32)
W[0, :, :] = W1
W[1, :, :] = W2
print(W)

W1_xs = W[0][:, 0]
W1_ys = W[0][:, 1]
plt.scatter(W1_xs, W1_ys, color='blue')
W2_xs = W[1][:, 0]
W2_ys = W[1][:, 1]
plt.scatter(W2_xs, W2_ys, color='gold')

plt.grid()
plt.show()


# Train when T1 is identity:

# In[46]:

T1, T, A = train(W)
print_results(W, T1, T, A)


# Train when T1 is not identity either:

# In[47]:

T1, T, A = train(W, t1_identity=False)
print_results(W, T1, T, A)


# #### 3 language - 3 words example
# 
# Diagram:  
# <span style="color:blue">**W1 points**</span>  
# <span style="color:gold">**W2 points**</span>  
# <span style="color:red">**W3 points**</span>  

# In[48]:

W1 = np.array([[1, 0], [0, 1], [1, 1]]).astype(np.float32)
W2 = np.array([[2, 2], [1, -1], [-1, -1]]).astype(np.float32)
W3 = np.array([[1.5, 1.5], [1.2, 0.6], [-1.2, -1.5]]).astype(np.float32)


W = np.ndarray(shape=(3,3,2), dtype=np.float32)
W[0, :, :] = W1
W[1, :, :] = W2
W[2, :, :] = W3
print(W)

W1_xs = W[0][:, 0]
W1_ys = W[0][:, 1]
plt.scatter(W1_xs, W1_ys, color=global_colors[0])
W2_xs = W[1][:, 0]
W2_ys = W[1][:, 1]
plt.scatter(W2_xs, W2_ys, color=global_colors[1])
W3_xs = W[2][:, 0]
W3_ys = W[2][:, 1]
plt.scatter(W3_xs, W3_ys, color=global_colors[2])

plt.grid()
plt.show()


# Train when T1 is identity:

# In[49]:

T1, T, A = train(W, learning_rate=0.007, num_steps=3001)
print_results(W, T1, T, A)


# Train when T1 is not identity either:

# In[50]:

T1, T, A = train(W, learning_rate=0.007, num_steps=3001, t1_identity=False)
print_results(W, T1, T, A)

