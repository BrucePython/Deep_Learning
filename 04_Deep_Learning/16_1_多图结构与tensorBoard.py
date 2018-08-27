# 一个程序里可以存在多张图，但程序很少使用多图结构

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 创建一张图
g = tf.Graph()
# 使用这张图，在g图例操作
with g.as_default():
    # 在新定义的图里，进行操作
    con_g = tf.constant(4.0)
    print(con_g.graph)


con1 = tf.constant(11.0, name="con1")
con2 = tf.constant(12.0, name="con2")
sum_ = tf.add(con1,con2)

print(tf.get_default_graph())   # 默认图
print(g)    # 新建图

with tf.Session() as sess:
# with tf.Session(graph=g) as sess:   # 会报错，因为sum_不在g图里
    print(sess.run(sum_))
    # print(g)
    # 在会话当中序列化图到events文件，然后启动TensorBoard
    # tf.summary.FileWriter("./", graph=sess.graph)
    # 然后启动TensorBoard， 启动命令：tensorboard --logdir="./"
