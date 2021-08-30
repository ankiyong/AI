# TensroFlow
## linear regression

```python
import tensorflow.compat.v1 as tf

X = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
H = W * X + b

loss = tf.reduce_mean(tf.square(H-y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10000
for step in range(epochs):
    _,loss_val,W_val,b_val = sess.run([train,loss,W,b],feed_dict={X:[1,2,3,4],y:[3,5,7,9]})
    if step%2000 == 0:
        print(loss_val,W_val,b_val)
print(sess.run(H,feed_dict={X:[10,11,12,13]}))
```

