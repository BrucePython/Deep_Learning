import tensorflow as tf

x = tf.constant([[True, True], [True, False]])

# 判断每行是否全为True/False
y = tf.reduce_all(x, 1)  # [False,False]

with tf.Session() as sess:
    print(sess.run(y))