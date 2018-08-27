# 实现加法
# session 相当于开启了程序与C++接口之间，分配资源或进行运算的连接
import tensorflow as tf

# 1. 构建图阶段：图（程序）的定义
con1 = tf.constant(11.0, name="con1")
con2 = tf.constant(12.0, name="con2")

sum_ = tf.add(con1, con2, name="add_1")

"""
with tf.device("GPU:0")   # 在GPU里运行图：
    sum_ = tf.add(con1, con2, name="add_1")
"""

a = 1
b = 2
c = a + b   # 由于c不是op类型，所以非tensor，不能run
c = a + con1    # 由于con1是op类型，c经过重载，可以run


# 这个数据，在图中没有明确定义好数据的内容（参数：指定类型，指定形状）
plt = tf.placeholder(tf.float32, shape=[None,2])    # 固定两列，可以自定义行数

# 2. 执行图阶段：会话去运行图（程序）
# 会话运行的是默认图，只能运行默认图里面的OP。可通过graph参数改变会话的图
# tf中，只能指定会话（会话使用上下文环境）去运行。只能在会话里面运行该图的程序；打印变量无所谓
with tf.Session() as sess:
# 显示硬件配置信息
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    print(sess.run(sum_))
    # 使用tf.operation.eval()也可运行operation
    print(sum_.eval())
    # run参数1【fetch】: 运行多个，使用列表
    print(sess.run([con1,con2,sum_]))
    # print(sess.run(c))                # c的重载
    # run参数2【feed_dict】：运行时候提供数据，一般不确定数据形状时，可以结合placeholder去使用。
    # 用在训练的时候实时提供要训练的批次数据
    print(sess.run(plt, feed_dict={plt:[[1,2],[3,4],[5,6]]}))
