# TensroFlow
## multi_classification

```python
import tensorflow.compat.v1 as tf
train_X = [
    [10,7,8,5],
    [8,8,9,4],
    [7,8,2,3],
    [6,3,9,3],
    [7,5,7,4],
    [3,5,6,2],
    [2,4,3,1]
]

train_y = [
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,0,1]
]

X = tf.placeholder(shape=[None,4],dtype=tf.float32)
y = tf.placeholder(shape=[None,3],dtype=tf.float32)

W = tf.Variable(tf.random_normal([4,3]),name='weight')
b = tf.Variable(tf.random_normal([3]),name='bias')
logit = tf.matmul(X,W)+b
H = tf.nn.softmax(logit)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10000
for step in range(epochs):
    _,loss_val,W_val,b_val = sess.run([train,loss,W,b],feed_dict={X:train_X,y:train_y})
    if step%2000 == 0:
        print(loss_val,W_val,b_val)
print(sess.run(H,feed_dict={X:[[5,9,8,2]]}))
```

