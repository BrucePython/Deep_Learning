import tensorflow as tf

# 4个英文字母的二维码识别，每个字母的输出结果为[0,0,0,0,...,0,0],26个元素组成，1为对应的字母的位置
"""
步骤：

"""


class Captcha_Recognize(object):
    def __init__(self):
        self.height = 20
        self.width = 80
        self.channel = 3
        # 每个验证码的目标值个数（4个字符）
        self.label_num = 4
        # 每个目标值有26种可能性
        self.possibility = 26
        self.train_batch_size = 100

    def read_tfrecord(self):
        # 1. 构建文件队列(列表）= 路径+文件名
        file_queue = tf.train.string_input_producer(["./captcha.tfrecords"])

        # 2. 读取TFRecord数据，解析example协议
        reader = tf.TFRecordReader()
        _, value = reader.read(file_queue)  # 默认读取一个样本

        # 解析example协议, feature 相当于字典
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string)
        })

        # 3. 解码
        image = tf.decode_raw(feature["image"], tf.uint8)
        label = tf.decode_raw(feature["label"], tf.uint8)

        # 确定形状和类型：
        # 图片形状[20,80,3]
        # 目标值形状 [4]
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        label_reshape = tf.reshape(label, [self.label_num])

        image_type = tf.cast(image_reshape, tf.float32)
        label_type = tf.cast(label_reshape, tf.int32)

        # 4. 批处理
        image_batch, label_batch = tf.train.batch([image_type, label_type],
                                                  batch_size=self.train_batch_size,
                                                  num_threads=1,
                                                  capacity=self.train_batch_size)

        print("1", image_batch, label_batch)
        return image_batch, label_batch

    # 定义专门初始化权重和偏置的两个函数
    @staticmethod
    def weight_initialize(shape):
        weight = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1), name="weight")
        return weight

    @staticmethod
    def bias_initialize(shape):
        bias = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1), name="bias")
        return bias

    def captcha_model(self, image_batch):
        """建立一层全连接网络输出，每次传入100（self.train_batch_size）张图片"""
        # 建立模型就是为了输出预测目标值，所以不需要传入真实目标值
        with tf.variable_scope("captcha_fc_model"):

            # 特征值形状：4维[100, 20, 80, 3] --> 2维[100, 20 * 80 * 3]，为了矩阵运算
            image_batch_reshape = tf.reshape(image_batch, [self.train_batch_size, 20 * 80 * 3])

            # x[100,20*80*3] * w[20*80*3,26*4] + b[104] = y[100,26*4]
            weight = self.weight_initialize([self.height * self.width * self.channel, self.label_num * self.possibility])
            bias = self.bias_initialize([self.label_num * self.possibility])

            y_predict = tf.matmul(image_batch_reshape, weight) + bias
        return y_predict

    def turn_to_onehot(self, label_batch):
        """真实值转换成one-hot编码，[100,4] --> [100,26*4]"""
        # 每一个目标值的类别数：每个输出字母能有多少种可能性
        with tf.variable_scope("one_hot"):
            # [100,4] --> [100,4,26]
            y_true = tf.one_hot(label_batch, depth=self.possibility, on_value=1.0)

        return y_true

    def loss(self, y_true, y_predict):
        """建立验证码 四个目标值的总损失"""
        with tf.variable_scope("loss"):
            y_true_reshape = tf.reshape(y_true, [self.train_batch_size, self.label_num * self.possibility])
            # 先进行softmax计算概率，然后计算交叉熵损失
            all_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_reshape, logits=y_predict, name="compute_loss")

            # 求出平均损失
            loss = tf.reduce_mean(all_loss)

        return loss

    def sgd(self, loss):
        """梯度下降优化损失"""
        with tf.variable_scope("sgd"):
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
        return train_op

    def acc(self, y_true,y_predict):
        """计算准确率"""
        with tf.variable_scope("acc"):
            # 统一使用3维数组[100,4,26],y_predict [100,104] --> [100,4,26]
            y_predict_reshape = tf.reshape(y_predict, [100, self.label_num, self.possibility])

            # 求出最大值对应的角标
            y_true_max_index = tf.argmax(y_true,2)
            y_predict_reshape = tf.argmax(y_predict_reshape,2)

            # 判断【真实最大值】的【角标】与【预测最大值】的【角标】是否相等的列表
            equal_list = tf.equal(y_true_max_index, y_predict_reshape)

            # 判断每行是否全为True/False
            judge_row = tf.reduce_all(equal_list,1)

            # 计算平均准确率
            accuracy = tf.reduce_mean(tf.cast(judge_row,tf.float32))

        return accuracy



    def train(self):
        """模型训练逻辑"""
        # 1. 通过接口，获取特征值和目标值: image_batch[100, 20, 80, 3], label_batch[100,4]
        image_batch, label_batch = self.read_tfrecord()

        # 2. 建立验证码识别的模型，y_predict
        y_predict = self.captcha_model(image_batch)     # shape: [100,104]

        # 转换label_batch到one-hot编码
        y_true = self.turn_to_onehot(label_batch)       # shape: [100,4,26]

        # 3. 利用真实值和预测值建立交叉熵损失
        loss = self.loss(y_true, y_predict)

        # 4. 对损失进行梯度下降优化
        train_op = self.sgd(loss)

        # 5. 计算准确率
        accuracy = self.acc(y_true, y_predict)


        # 会话训练：
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 生成线程管理器
            coord = tf.train.Coordinator()

            # 指定开启子线程，读取数据
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)    # 返回：线程开启的数量


            for i in range(1000):
                _, accuracy_run = sess.run([train_op,accuracy])
                # 子线程每批次读取100张图片，image_batch已经在模型里，不需要使用feed_dict获取数据
                print("第【%d】步，准确率为【%f】" % (i, accuracy_run))


            # 回收线程（因为有数据读取）
            coord.request_stop()
            coord.join(threads=threads)

cr = Captcha_Recognize()
cr.train()
