# 一层全连接，手写数字输出10个类型，也就最后一层有10个神经元

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import os

# 定义一个是否训练或预测的标志（全局变量）
tf.app.flags.DEFINE_integer("is_train", 1, "训练or预测")
FLAGS = tf.app.flags.FLAGS


def full_connected_nn():
    """全连接层神经网络进行Mnist手写数字识别训练"""

    # 1. 获取数据，定义特征值和目标值
    mnist = input_data.read_data_sets("./mnist", one_hot=True)
    # print(mnist.train.labels.shape)
    # print(mnist.train.images.shape)
    # images, labels = mnist.train.next_batch(10)

    with tf.variable_scope("data"):

        # 定义特征值占位符，为了实时给x传递数据，所以先用占位符表示；同理y_true
        x = tf.placeholder(tf.float32, shape=[None, 784], name="feature")
        y_true = tf.placeholder(tf.float32, shape=[None, 10], name="label")

    # 2. 根据识别的分类个数（10个），建立全连接层网络。（如多隐层，则在此步骤处理）
    with tf.variable_scope("fc_model"):

        # 使用变量OP，随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="weight")
        bias = tf.Variable(tf.random_normal([10], mean=0.0, stddev=1.0), name="bias")

    # 进行全连接层的矩阵运算，x[None,784] * w[784,10] + b[10] = y_predict[None,10]
    y_predict = tf.matmul(x, weight) + bias



    # 3. 根据输出的结果计算其softmax，并与真实值进行交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):

        # 先进行网络输出值的softmax概率计算，再计算每个样本的损失
        all_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict, name="compute_loss")

        # 求出平均损失
        loss = tf.reduce_mean(all_loss)

    # 4. 使用梯度下降优化器，优化损失
    with tf.variable_scope("GD"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=loss)
        # 使用train_op去训练

    # 5. 求出每次训练的准确度
    # 求出每个样本的列表：真实值与预测值是否在同一个位置
    with tf.variable_scope("accuracy"):
        # 求出每个样本的列表
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # 计算样本相等的比例
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 6.1 收集要在Tensorboard观察的张量值
    tf.summary.scalar("loss", loss)             # 数值型（scalar）:loss，accuracy
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("weight",weight)       # 高维度型（histogram）:weight，bias
    tf.summary.histogram("bias",bias)

    # 6.2 合并变量
    merged = tf.summary.merge_all()

    # 7.1 创建保存模型的OP
    saver = tf.train.Saver()

    # 0. 开启会话进行训练（train_op）
    with tf.Session() as sess:

        # 初始化变量OP
        sess.run(tf.global_variables_initializer())

        # 6.3 创建event文件
        file_writer = tf.summary.FileWriter("./mnist", graph=sess.graph)

        # 7.3 加载本地模型继续训练 或 拿来进行预测【测试集】
        ckpt = tf.train.latest_checkpoint("./mnist")    # 重新加载模型，加载的数值都放入会话sess里

        # 7.4 判断模型是否存在
        if ckpt:
            saver.restore(sess, ckpt)

        # 8.1 进行训练
        if FLAGS.is_train == 1:

            # 循环训练
            for i in range(2000):    # 步数=1000
                # 每批次给50个样本
                x_mnist, y_mnist = mnist.train.next_batch(50)
                # run的feed_dict机制，指定给占位符变量传数据。只要运行的变量中含有占位符，都要进行feed_dict
                _, loss_run, accuracy_run, merged_run = sess.run([train_op, loss, accuracy, merged], feed_dict={x:x_mnist, y_true:y_mnist})

                print("第【%d】步，50个误差为【%f】，准确率为【%f】" % (i,loss_run,accuracy_run))

                # 6.4 写入运行的结果，到文件里
                file_writer.add_summary(merged_run,i)

                # 7.2 每隔100步，保存一次模型的参数
                if i % 100 == 0:
                    saver.save(sess, "./mnist/fc_nn_model")

        # 8.2 进行预测
        else:
            # 预测100个样本：还是用前面的模型预测，只不过参数是已加载的模型的参数
            for i in range(100):
                image, label = mnist.test.next_batch(1)

                print("第【%d】步，图片的真实数字为【%d】，神经网络预测的数字为【%d】" %(
                    i,
                    # tf.argmax(sess.run(label, feed_dict={x: image, y_true:label}), 1).eval(), # label是一个值，不是tensor。求最大值之后变成了tensor，然后用eval取值
                    tf.argmax(label,1).eval(),  # 对label求最大值，然后取值
                    tf.argmax(sess.run(y_predict, feed_dict={x: image, y_true:label}), 1).eval()
                ))


if __name__ == '__main__':
    full_connected_nn()
