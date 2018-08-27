import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 加入命令行参数：
# 训练步数
tf.app.flags.DEFINE_integer("train_step", 0, "训练模型的步数") # train_step：步数，初始为0
# 定义模型的路径
tf.app.flags.DEFINE_string("model_dir", " ", "模型保存的路径") # model_dir：传递的路径+文件名，初始为" "
# 定义获取命令行参数
FLAGS = tf.app.flags.FLAGS

# 2. 使用命令行运行：
# $ python 19_线性回归训练案例.py --train_step=500 --model_dir="./model/myregression"

class MylinearRegression():
    """实现线性回归"""

    def __init__(self):
        # 自定义学习率；学习率越大，损失也越大，最总导致梯度紊乱（损失、参数优化成nan）
        # 梯度爆炸：权重无穷大；梯度消失：w无穷小
        self.learning_rate = 0.1

    # 1. 获取数据
    def input(self):
        """获取需要训练的数据"""
        # x_data = [100,1] ; y_true = x*0.7 + 0.8
        x_data = tf.random_normal(shape=[100, 1], mean=0.0, stddev=1.0,
                                  name="x_data")  # 【进行矩阵相乘，必须是二维结构】; 100个样本，每个样本一个特征
        # 进行矩阵运算: [100,1]*[1,1] = [100,1]；[[0.7]]为二维数组，形状（1，1）
        y_true = tf.matmul(x_data, [[0.7]]) + 0.8
        return x_data, y_true

    # 2. 建立模型
    def inference(self, feature):
        """根据数据的特征值，建立线性回归模型
        :param feature: 数据特征值，[100,1]
        :return: y_predict
        """
        # 定义一个命名空间（不是为了共享变量，而是为了这一块东西都在同一个空间里面）
        with tf.variable_scope("linear_model"):
            # 随机初始化权重和偏置
            # 权重和偏置必须使用tf.Variable去定义，因为只有Variable才能被梯度下降所训练
            # 权重的形状必须是[1,1]；如果是由n个特征，则权重的形状为[n,1]
            # trainable=False   # 指定是否训练这个参数
            self.weight = tf.Variable(tf.random_normal(shape=[1, 1], mean=0.0, stddev=1.0), name="weight")
            self.bias = tf.Variable(tf.random_normal(shape=[1], mean=0.0, stddev=1.0), name="bias")

            # 建立模型预测
            y_predict = tf.matmul(feature, self.weight) + self.bias

        return y_predict

    # 3. 求出损失
    def loss(self, y_true, y_predict):
        """根据真实值和测试值，计算出均方误差"""
        # 定义一个命名空间
        with tf.variable_scope("losses"):
            # 求均方误差损失：把100个样本 先sum((y_true-y_predict)^2)， 再求平均值mean()，得出每个样本的平均误差
            # 求出损失：均方误差
            # tf.reduce_mean() 对列表中的数据求和之后，再求出平均值
            loss = tf.reduce_mean(tf.square(y_true - y_predict))

        return loss

    # 4. 梯度下降
    def sgd_op(self, loss):
        """利用梯度下降优化器去优化损失（优化模型参数）
        :param loss: 损失大小
        :return: 梯度下降OP，（注意不是返回结果，OP操作需要在会话里面运行）
        """
        # 定义一个命名空间，使代码结构更加清晰，Tensorboard图结构清楚
        with tf.variable_scope("train_op"):
            # .minimize() 去优化损失
            train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        return train_op


    def merge_summary(self, loss):
        """定义收集张量的函数"""
        # 收集对于损失函数和准确率等单变量值
        tf.summary.scalar("loss", loss)     # 单值变量

        # 收集高纬度张量值，2阶3阶等
        tf.summary.histogram("w", self.weight)
        tf.summary.histogram("b", self.bias)

        # 合并变量（OP）,OP不能放到序列化文件，是把OP的值
        merged = tf.summary.merge_all()
        return merged

    def train(self):
        """专门用于训练的函数"""
        # 此时图已经定义完毕，但需要将每部分连起来，同时需要用会话运行图
        # 获取默认的图
        g = tf.get_default_graph()

        # 在默认图当中做操作
        with g.as_default():
            # 进行训练
            # 1. 获取数据
            x_data, y_true = self.input()

            # 2. 利用模型，得出预测结果
            y_predict = self.inference(x_data)

            # 3. 损失计算
            loss = self.loss(y_true, y_predict)

            # 4. 优化损失
            train_op = self.sgd_op(loss)

            # 5. 收集要观察的张量
            merged = self.merge_summary(loss)

            # 6. 定义一个保存文件的saverOP
            saver = tf.train.Saver()

            # 开启会话去训练数据
            with tf.Session() as sess:
                # 初始化变量
                sess.run(tf.global_variables_initializer())

                # 创建events文件，然后用Tensorboard启动
                file_writer = tf.summary.FileWriter("./model", graph=sess.graph)

                # 打印模型在训练之前的初始化参数
                # 由于weight和bias是op参数，所以用.eval() 拿出变量的值，而不是拿出变量op
                print("模型初始化的参数权重： %f， 偏置为：%f" % (self.weight.eval(), self.bias.eval()))

                # 加载模型，从模型中找出与当前训练的模型代码中【名字一样的OP操作weight/bias】，覆盖原来的值
                ckpt = tf.train.latest_checkpoint("./model")    # ckpt就是（路径+模型名字）
                # 判断模型是否存在
                if ckpt:
                    saver.restore(sess,ckpt)    # 重新加载模型，加载的数值都放入会话sess里
                    # 两个print放if里面和外面都一样。
                    print("【第一次】加载保存的模型，权重： %f， 偏置为：%f" % (self.weight.eval(), self.bias.eval()))
                    print("以模型当中的参数继续去训练")
                # 运行最后一步，tf的机制：从前往后
                for i in range(FLAGS.train_step):
                    # train_op返回的是结果, 不用接受，所以_
                    _, summary = sess.run([train_op, merged])  # 默认只运行一次
                    # 把张量的值（summary）写入到events文件当中
                    file_writer.add_summary(summary,i)        # i 指定第几次写入
                    # 预期值：w=0.7, b=0.8
                    print("第【%d】步，总损失变化【%f】，模型优化后的参数权重【%f】，偏置为【%f】" % (i, loss.eval(),self.weight.eval(), self.bias.eval()))
                    # 每隔100步，保存一次模型（值）
                    if i % 100 == 0:
                        # 要指定路径+文件名
                        saver.save(sess, FLAGS.model_dir)

if __name__ == '__main__':
    lr = MylinearRegression()
    lr.train()
