# tensorflow里获取的是op名字的值（name=“xxx”），而不是op变量名字的值

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 特殊的创建张量OP:
# 1. 必须手动初始化(赋值)：
# 在使用初始化之前，一定要先进行初始化：
var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0),name="var_name")
# init_varop = tf.global_variables_initializer()    # 直接在会话里运行

# 2. 重新赋值一个新的值，原来的值也会改变
new_var = var.assign([[1,2,3],[4,5,6]])
# 修改原来的值，并添加新的值
new_var_2 = var.assign_add([[1,2,3],[4,5,6]])

# 3. 命名空间(variable_scope): 会在op名字前面加上空间的名字"my_scope/con1:0"
with tf.variable_scope("my_scope"):
    con1 = tf.constant(11.0, name="con1")
    con2 = tf.constant(12.0, name="con1")

# 4. 共享变量(variable_scope + AUTO_REUSE + get_variable)：设置可重复使用的OP名字，相当于全局变量，name全局唯一
with tf.variable_scope("my_scope1", reuse=tf.AUTO_REUSE):
    con3 = tf.get_variable(initializer=tf.random_normal([2,3],mean=0.0,stddev=1.0), name="con_global")
    con4 = tf.get_variable(initializer=13.0, name="con_global")
    print(con3)
    print(con4)

# 开启会话
with tf.Session() as sess:
    # 运行OP操作，才会生效
    # sess.run(init_varop)
    sess.run(tf.global_variables_initializer())   # 运行初始化OP
    # print(sess.run([new_var_2, var]))
    # print(con1)
    # print(con2)
    # print(sum1)
    print(con3.eval())
    print(con4.eval())
    # print(sum2)

    tf.summary.FileWriter("./", graph=sess.graph)