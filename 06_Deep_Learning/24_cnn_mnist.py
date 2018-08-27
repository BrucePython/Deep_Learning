"""
卷积模型（确定网络结构和参数）：
两层卷积池化和一层输出层

第一层
    卷积：32个filter、大小5*5、strides=1、padding="SAME"
    激活：Relu
    池化：大小 2x2、strides = 2
第二层
    卷积：64个filter、大小5*5、strides=1、padding="SAME"
    激活：Relu
    池化：大小 2x2、strides = 2
全连接层

重点：计算每层数据的变化

梯度爆炸：
1. 调整参数w,b的值
2. 使用tf.train.AdamOptimizer()
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 定义专门初始化权重和偏置的两个函数（因为两个参数的形状不同，所以要把权重和偏置分开）
def weight_initialize(shape):
    weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1), name="weight")
    return weight


def bias_initialize(shape):
    bias = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1), name="bias")
    return bias


def cnn_model():
    """自定义卷积模型"""
    # 1. 定义特征值和目标值的占位符，便于卷积计算:x[None,784], y[None,10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784], name="feature")
        y_true = tf.placeholder(tf.float32, [None, 10], name="y_true")

    # 2. 第一层：
    with tf.variable_scope("layer_1"):
        # 2.1 卷积层：32个filter、大小5*5、strides=1、padding="SAME"

        # 初始化权重[5,5,1,32] 和 偏置[32]
        conv1_weight = weight_initialize([5, 5, 1, 32])
        conv1_bias = bias_initialize([32])

        # 特征形状变成4维，用于卷积运算。（重点：None就是-1）
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # 卷积运算: [-1,28,28,1] --> [-1,28,28,32]
        conv1_x = tf.nn.conv2d(input=x_reshape,
                               filter=conv1_weight,
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               name="conv1") + conv1_bias
        # 2.2 激活层运算
        relu1_x = tf.nn.relu(conv1_x, name="relu1")

        # 2.3 池化层: 大小 2x2、strides=2、padding="SAME"
        # [-1,28,28,32] --> [-1,14,14,32]
        pool1_x = tf.nn.max_pool(value=relu1_x,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool1")

    # 3. 第二层
    with tf.variable_scope("layer_2"):
        # 3.1 卷积层：64个filter、大小5*5、strides=1、padding="SAME"

        # 初始化权重[5,5,32,64] 和 偏置[64]
        conv2_weight = weight_initialize([5, 5, 32, 64])
        conv2_bias = bias_initialize([64])

        # 卷积运算：[-1,14,14,32] --> [-1,14,14,64]
        conv2_x = tf.nn.conv2d(input=pool1_x,
                               filter=conv2_weight,
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               name="conv1") + conv2_bias

        # 3.2 激活层运算
        relu2_x = tf.nn.relu(conv2_x, name="relu2")

        # 2.3 池化层: 大小 2x2、strides=2、padding="SAME"
        # [-1,14,14,64] --> [-1,7,7,64]
        pool2_x = tf.nn.max_pool(value=relu2_x,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool2")

    # 4. 全连接层
    with tf.variable_scope("last_layer"):
        # 初始化权重[7*7*64,10] 和 偏置[64]
        fc_weight = weight_initialize([7 * 7 * 64, 10])
        fc_bias = bias_initialize([10])

        # 4维转换2维
        pool2_x_reshape = tf.reshape(pool2_x, [-1, 7 * 7 * 64])

        # 全连接层矩阵运算
        y_predict = tf.matmul(pool2_x_reshape, fc_weight) + fc_bias

    return x, y_true, y_predict


def cnn_mnist():
    """卷积网络识别训练"""

    # 1. 获取数据
    mnist = input_data.read_data_sets("./mnist", one_hot=True)

    # 2. 建立卷积网络模型：（说明）
    x, y_true, y_predict = cnn_model()

    # 3. 根据输出的结果计算其softmax，并与真实值进行交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):
        # 先进行网络输出值的softmax概率计算，再计算每个样本的损失
        all_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict, name="compute_loss")

        # 求出平均损失
        loss = tf.reduce_mean(all_loss)

    # 4. 梯度下降优化
    with tf.variable_scope("GD"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=loss)
        # 使用adam优化器：
        # train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

    # 5. 计算准确率
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))


        # 0. 开启会话进行训练（train_op）
        with tf.Session() as sess:
            # 初始化变量OP
            sess.run(tf.global_variables_initializer())

            # 循环训练
            for i in range(2000):  # 步数=1000
                # 每批次给50个样本
                x_mnist, y_mnist = mnist.train.next_batch(50)
                # run的feed_dict机制，指定给占位符变量传数据。只要运行的变量中含有占位符，都要进行feed_dict
                _, loss_run, accuracy_run= sess.run([train_op, loss, accuracy],
                                                                 feed_dict={x: x_mnist, y_true: y_mnist})

                print("第【%d】步，50个误差为【%f】，准确率为【%f】" % (i, loss_run, accuracy_run))

    return None


if __name__ == '__main__':
    cnn_mnist()
