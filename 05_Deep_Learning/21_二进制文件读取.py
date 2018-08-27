import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CifarRead(object):
    """读取cifar10类别的二进制文件"""

    def __init__(self):
        # 每个图片的样本属性：
        self.height = 32
        self.width = 32
        self.channel = 3

        # bytes:
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        # 一个样本的总字节数 3073
        self.all_bytes = self.label_bytes + self.image_bytes

    def bytes_read(self, file_list):
        """
        读取二进制解码张量
        :param file_list: 路径+文件名的列表，（是一个1阶张量）
        """
        # 1. 构造文件队列
        file_queue = tf.train.string_input_producer(file_list)

        # 2. 利用【二进制】读取器去【读取】文件队列的内容
        read = tf.FixedLengthRecordReader(self.all_bytes)  # 默认必须指定读取一个样本
        key, value = read.read(file_queue)

        # 3. 对二进制数据进行【解码】; shape：从（）变成 （?,?,?）; dtype: 从string 变成 uint8
        label_image = tf.decode_raw(value, tf.uint8)
        # 为了训练方便，一般会把特征值和目标值分开处理
        print(label_image)

        # 使用tf.slice进行切片
        label = tf.slice(label_image, [0], [self.label_bytes])  # 目标值
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])  # 特征值
        print(label, image)

        # 4. 处理类型和图片的形状
        label = tf.cast(label, tf.int32)
        # 图片形状 32*32*3
        # reshape(3072,) ---> [channel,height,width]
        depth_major = tf.reshape(image, [self.channel, self.height, self.width])
        print(depth_major)
        # transpose [channel,height,width] ---> [height,width,channel] 要求需要用这个格式
        image_reshape = tf.transpose(depth_major, [1, 2, 0])  # 把维度进行互换，否则会出错
        print(image_reshape)

        # 5. 批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

    def write_to_TFRecordWriter(self, image_batch, label_batch):
        """将数据写入TFRecord存储器"""
        # 构造TFRecord存储器
        writer = tf.python_io.TFRecordWriter("./cifar.tfrecords")
        # 循环将每一个样本构造成一个example，然后序列化写入
        for i in range(10):
            # 取出相应的第i个样本的特征值和目标值
            # 写入的是具体的张量的值，不是OP的名字
            image = image_batch[i].eval().tostring()  # tensor转bytes(bytes需要tostring()序列化)
            label = label_batch[i].eval()[0]  # tensor转int
            # example模板：
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            # 序列化写入每个样本的example
            writer.write(example.SerializeToString())

        # 文件需要关闭
        writer.close()
        return None


if __name__ == '__main__':
    # 指定文件读取的路径
    file_name = os.listdir("./cifar10/")  # 返回的仅仅只是文件名
    # 拼接路径+文件名, 过滤去处非bin文件, file[-3:]后三位
    file_list = [os.path.join("./cifar10/", file) for file in file_name if file[-3:] == "bin"]

    cr = CifarRead()

    image_batch, label_batch = cr.bytes_read(file_list)
    # 输出4D张量：Tensor("batch:0", shape=(10, 200, 200, 3), dtype=float32)

    # 把图片打印出来
    with tf.Session() as sess:
        # 创建线程回收的协调员
        coord = tf.train.Coordinator()

        # 需要手动开启子线程去进行批处理读取到队列操作
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run([image_batch, label_batch]))

        # 存入数据
        cr.write_to_TFRecordWriter(image_batch, label_batch)
        # 回收线程
        coord.request_stop()
        coord.join(threads)
