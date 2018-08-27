# TensorFlow的张量具有两种形状变换，动态形状(tf.reshape)和静态形状(tf.set_shape)
"""
关于动态形状和静态形状必须符合以下规则:

静态形状:
转换静态形状的时候，1阶到1阶，2阶到2阶，【不能跨阶数改变形状】
对于已经固定的张量的静态形状的张量，不能再次设置静态形状；如果形状不固定，可以修改张量的形状
修改的是本身的形状

动态形状:
tf.reshape()动态创建新张量时，张量的元素个数必须匹配
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

con1 = tf.constant([[1,2],[3,4]])

# 静态形状：
# con1.set_shape([4])             # 改成1阶的形状（4个数）。错误操作：无法匹配
# 虽然都是一排有4个数，但是 (4,)是1阶 与 (1,4)是2阶

plt = tf.placeholder(tf.float32, [None,4])
# 对于没有固定形状的shape，可以使用set_shape。None可变，4恒定
plt.set_shape([5, 4])
print(plt)


# 动态形状:
# 使用tf.reshape创建了新的张量，并没有修改原来的张量
# reshape之前和之后的【元素个数】，保持不变：
# 【5行4列】变成【4行5列】，【2行2列】变成【1行4列】
plt_reshape = tf.reshape(plt,[4,5])
print(plt, plt_reshape)
con1_reshape = tf.reshape(con1,[1,4])
print(con1, con1_reshape)