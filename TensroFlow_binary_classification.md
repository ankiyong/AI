# Tensorflow
## binary classification

import tensorflow.compat.v1 as tf
##### Data
train_X = [
    [1,0],
    [2,0],
    [5,1],
    [2,3],
    [3,3],
    [8,1],
    [10,0]

]

train_y = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
]

X = tf.placeholder(shape=[None,2],dtype=tf.float32)
y = tf.placeholder(shape=[None,1],dtype=tf.float32)

W = tf.Variable(tf.random_normal([2,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
logit = tf.matmul(X,W)+b
H = tf.sigmoid(logit)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=y))
train = tf.train.GradientDescentOptimizer(0.0004).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10000
for step in range(epochs):
  _,loss_val,W_val,b_val = sess.run([train,loss,W,b],feed_dict={X:train_X,y:train_y})
  if step%1000 == 0:
    print(loss_val,W_val,b_val)

print(sess.run(H,feed_dict={X:[[5,6]]}))
