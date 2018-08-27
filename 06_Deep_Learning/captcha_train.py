import tensorflow as tf


class CaptchaIdentification(object):
    """
    验证码的读取数据、网络训练
    """
    def __init__(self):

        # 验证码图片的属性
        self.height = 20
        self.width = 80
        self.channel = 3
        # 每个验证码的目标值个数(4个字符)
        self.label_num = 4
        self.feature_num = 26

        # 每批次训练样本个数
        self.train_batch = 100

    @staticmethod
    def weight_variables(shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1))
        return w

    @staticmethod
    def bias_variables(shape):
        b = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1))
        return b

    def read_captcha_tfrecords(self):
        """
        读取验证码特征值和目标值数据
        :return:
        """
        # 1、构造文件的队列
        file_queue = tf.train.string_input_producer(["./tfrecords/captcha.tfrecords"])

        # 2、tf.TFRecordReader 读取TFRecords数据
        reader = tf.TFRecordReader()

        # 单个样本数据
        key, value = reader.read(file_queue)

        # 3、解析example协议
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string)
        })

        # 4、解码操作、数据类型、形状
        image = tf.decode_raw(feature["image"], tf.uint8)
        label = tf.decode_raw(feature["label"], tf.uint8)

        # 确定类型和形状
        # 图片形状 [20, 80, 3]
        # 目标值 [4]
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        label_reshape = tf.reshape(label, [self.label_num])

        # 类型
        image_type = tf.cast(image_reshape, tf.float32)
        label_type = tf.cast(label_reshape, tf.int32)

        # 5、 批处理
        # print(image_type, label_type)
        # 提供每批次多少样本去进行训练
        image_batch, label_batch = tf.train.batch([image_type, label_type],
                                                   batch_size=self.train_batch,
                                                   num_threads=1,
                                                   capacity=self.train_batch)
        print(image_batch, label_batch)
        return image_batch, label_batch

    def captcha_model(self, image_batch):
        """
        建立全连接层网络
        :param image_batch: 验证码图片特征值
        :return: 预测结果
        """
        # 全连接层
        # [100, 20, 80, 3] --->[100, 20 * 80 * 3]
        # [100, 20 * 80 * 3] * [20 * 80 * 3, 104] + [104] = [None, 104] 104 = 4*26
        with tf.variable_scope("captcha_fc_model"):
            # 初始化权重和偏置参数
            self.weight = self.weight_variables([20 * 80 * 3, 104])

            self.bias = self.bias_variables([104])

            # 4维---->2维做矩阵运算
            x_reshape = tf.reshape(image_batch, [self.train_batch, 20 * 80 * 3])

            # [self.train_batch, 104]
            y_predict = tf.matmul(x_reshape, self.weight) + self.bias

        return y_predict

    def loss(self, y_true, y_predict):
        """
        建立验证码4个目标值的损失
        :param y_true: 真实值
        :param y_predict: 预测值
        :return: loss
        """
        with tf.variable_scope("loss"):
            # 先进行网络输出的值的概率计算softmax,在进行交叉熵损失计算
            # y_true:[100, 4, 26]------>[None, 104]
            # y_predict:[100, 104]
            y_reshape = tf.reshape(y_true,
                                   [self.train_batch, self.label_num * self.feature_num])

            all_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_reshape,
                                                               logits=y_predict,
                                                               name="compute_loss")
            # 求出平均损失
            loss = tf.reduce_mean(all_loss)

        return loss

    def turn_to_onehot(self, label_batch):
        """
        目标值转换成one_hot编码
        :param label_batch: 目标值 [None, 4]
        :return:
        """
        with tf.variable_scope("one_hot"):

            # [None, 4]--->[None, 4, 26]
            y_true = tf.one_hot(label_batch,
                                depth=self.feature_num,
                                on_value=1.0)
        return y_true

    def sgd(self, loss):
        """
        梯度下降优化损失
        :param loss:
        :return: train_op
        """
        with tf.variable_scope("sgd"):

            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        return train_op

    def acc(self, y_true, y_predict):
        """
        计算准确率
        :param y_true: 真实值
        :param y_predict: 预测值
        :return: accuracy
        """
        with tf.variable_scope("acc"):

            # y_true:[None, 4, 26]
            # y_predict：[None, 104]
            y_predict_reshape = tf.reshape(y_predict, [self.train_batch, self.label_num, self.feature_num])

            # 先对最大值的位置去求解
            euqal_list = tf.equal(tf.argmax(y_true, 2),
                     tf.argmax(y_predict_reshape, 2))

            # 需要对每个样本进行判断
            #  x = tf.constant([[True,  True], [False, False]])
            #  tf.reduce_all(x, 1)  # [True, False]
            accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(euqal_list, 1), tf.float32))

        return accuracy

    def train(self):
        """
        模型训练逻辑
        :return:
        """
        # 1、通过接口获取特征值和目标值
        # image_batch:[100, 20, 80, 3]
        # label_batch: [100, 4]
        # [[13, 25, 15, 15], [22, 10, 7, 10]]
        image_batch, label_batch = self.read_captcha_tfrecords()

        # 2、建立验证码识别的模型
        # 全连接层神经网络
        # y_predict [100, 104]
        y_predict = self.captcha_model(image_batch)

        # 转换label_batch 到one_hot编码
        # y_true:[None, 4, 26]
        y_true = self.turn_to_onehot(label_batch)

        # 3、利用真实值和目标值建立损失
        loss = self.loss(y_true, y_predict)

        # 4、对损失进行梯度下降优化
        train_op = self.sgd(loss)

        # 5、计算准确率
        accuracy = self.acc(y_true, y_predict)

        # 会话训练
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # 生成线程的管理
            coord = tf.train.Coordinator()

            # 指定开启子线程去读取数据
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 循环训练打印结果
            for i in range(1000):

                _, acc_run = sess.run([train_op, accuracy])

                print("第 %d 次训练的损失为：%f " % (i, acc_run))

            # 回收线程
            coord.request_stop()

            coord.join(threads)

        return None


if __name__ == '__main__':
    ci = CaptchaIdentification()
    ci.train()