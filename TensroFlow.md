# TensroFlow

import tensorflow as tf

1. 데이터 준비
X = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

2. 가설 설정
W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
H = W * X + b

3. loss function 설정
loss = tf.reduce_mean(tf.square(H-y))

4. 
train = tf.train.GradientDescentOptimizer(0.01).minize(loss)

5. sesstion 설정
sess = tf.Session()
sess.run(tf.global_variables_initializer())

6. 학습
epochs = 5000
for step in epochs:
  _,loss_val,W_val,b_val = sess.run([train,loss,W,b],feed_dict={X:[1,2,3,4],y:[3,5,7,9]})
  if step % 500 == 0:
    print("loss : {} \t W : {} \t b : {} ".format(loss,W,b)
7. 예측
print(sess.run(H,feed_dict={X:[10,11,12,13]})


