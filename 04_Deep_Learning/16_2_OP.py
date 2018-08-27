# 节点（OP）: 一般指的是运算操作

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 手动给OP命名，替代原OP的名字
con1 = tf.constant(11.0, name="con1")
con2 = tf.constant(12.0, name="con2")

sum_ = tf.add(con1, con2, name="add_1")
print(con1, con2, sum_)


# 2. 执行图阶段：会话去运行图（程序），会话运行的是默认的图，只能运行默认图里面的OP
with tf.Session() as sess:
    print(sess.run(sum_))
