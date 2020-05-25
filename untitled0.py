import tensorflow as tf
import numpy as np

tf.reset_default_graph() 

"""x_data = np.loadtxt('./inputData.txt', delimiter = ',', unpack = True, dtype = 'float32')
x_data = np.transpose(x_data)
y_data = np.loadtxt('./Label.txt', delimiter = ',', unpack = True, dtype = 'float32')
y_data = np.transpose(y_data)"""

# Features and Labels
X = tf.placeholder(tf.float32) # 1 x 7*3, (x, y, z)
Y = tf.placeholder(tf.float32) # 1 x 2, softmax 

# Network
w1 = tf.Variable(tf.random_normal([21, 128], stddev = .1))
b1 = tf.Variable(tf.zeros([128]))

L1 = tf.add(tf.matmul(X, w1), b1)
L1_act = tf.nn.relu(L1)
L1_do = tf.nn.dropout(L1_act, keep_prob = 0.7)
                 
w2 = tf.Variable(tf.random_normal([128, 128], stddev = .1))
b2 = tf.Variable(tf.zeros([128]))

L2 = tf.add(tf.matmul(L1_do, w2), b2)
L2_act = tf.nn.relu(L2)
L2_do = tf.nn.dropout(L2_act, keep_prob = 0.7)
      
w3 = tf.Variable(tf.random_normal([128, 2], stddev = .1))
b3 = tf.Variable(tf.zeros([2]))
         
m = tf.add(tf.matmul(L2_do, w3), b3)
model = tf.nn.softmax(m)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))

save_file = './train_model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
    
    print(sess.run(w1))