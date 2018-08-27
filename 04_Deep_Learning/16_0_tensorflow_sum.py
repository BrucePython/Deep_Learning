# 实现加法
# session 相当于开启了程序与C++接口之间，分配资源或进行运算的连接
import tensorflow as tf

# 1. 构建图阶段：图（程序）的定义
# 张量：TensorFlow中的基本数据对象
# 手动给OP命名，替代原OP的名字
con1 = tf.constant(11.0, name="con1")
con2 = tf.constant(12.0, name="con2")

# 节点（OP）: 一般指的是运算操作
sum_ = tf.add(con1, con2, name="add_1")
# print(con1, con2, sum_)


a = 1
b = 2
c = a + b   # 由于c不是op类型，所以非tensor，不能run
c = a + con1    # 由于con1是op类型，c进过重载，可以run

# 这个数据，在图中没有明确定义好数据的内容（参数：指定类型，指定形状）
plt = tf.placeholder(tf.float32, shape=[None,2])
# 不固定：shape=[None,4]



# 2. 执行图阶段：会话去运行图（程序），会话运行的是默认的图，只能运行默认图里面的OP
# tf中，只能指定会话（会话，会使用上下文环境）去运行。只能在会话里面运行程序；打印无所谓
with tf.Session() as sess:
# 打印硬件配置信息
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:

    # print(sess.run(sum_))
    print(sum_.eval())
    # print(sess.run([con1,con2,sum_]))               # 以列表形式打印数据

    # print(tf.get_default_graph())                   # 打印图结构，输出保存的内存地址
    # print(con1.graph)                               # 一样的内存地址
    # assert sum_.graph is tf.get_default_graph()     # 判断两者路径知否一致


    # print(sess.run(c))  # c的重载
    # print(sess.run(plt, feed_dict={plt:[[1,2],[3,4],[4,5]]}))  # c的重载
