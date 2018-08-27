# TensorFlow 的张量就是一个 n 维数组， 类型为tf.Tensor。Tensor具有以下两个重要的属性：type(数据类型), shape(形状(阶))
# 0维：()   1维-10个数：(10, )   2维-3行4列：(3, 4)   3维：(3, 4, 5)

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# tensor1 = tf.constant(4.0)
# tensor2 = tf.constant([1, 2, 3, 4])
# linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)

t1 = tf.zeros(shape=[3,4],dtype=tf.float32, name=None)

# 【类型转换】使用tf.cast()，将float32转成int32
t1_int = tf.cast(t1,dtype=tf.int32)
t2 = tf.ones(shape=[3,4],dtype=tf.float32, name=None)

# 随机创建一个三行四列的张量
tr = tf.random_normal(shape=[3,4],mean=0.0, stddev=1.0)


with tf.Session() as sess:
    print(t1.eval())
    print(t1_int.eval())
    # print(t2.eval())
    # print(tr.eval())


