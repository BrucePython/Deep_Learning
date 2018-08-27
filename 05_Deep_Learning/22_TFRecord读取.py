import tensorflow as tf


def read_TFRecord():
    """读取TFRecord文件"""
    # 1. 构建文件队列(列表）
    file_queue = tf.train.string_input_producer(["./cifar.tfrecords"])
    # 2. 读取TFRecord数据，并解析example协议
    reader = tf.TFRecordReader()
    # 默认只读取一个样本
    _, value = reader.read(file_queue)
    # 解析example协议, feature 相当于字典
    feature = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })

    # 3. 解码操作
    # 对于image是一个bytes类型，所以需要decode_raw去解码成uint8张量
    # 对于Label:本身是一个int类型，不需要去解码
    image = tf.decode_raw(feature["image"], tf.uint8)
    label = tf.cast(feature['label'], tf.int32)

    # 形状、类型: [32,32,3]-->bytes-->uint8、
    # 如果是按照RGB排列，则需要用transpose；否则不需要； 批处理之前，一定要进行形状固定
    image = tf.reshape(image, [32, 32, 3])

    # print(image)

    # 4. 批处理
    image_batch, label_batch = tf.train.batch([image, label], batch_size=10, num_threads=1, capacity=10)

    return image_batch, label_batch


if __name__ == '__main__':
    image_batch, label_batch = read_TFRecord()
    # 把图片打印出来
    with tf.Session() as sess:
        # 创建线程回收的协调员
        coord = tf.train.Coordinator()

        # 需要手动开启子线程去进行批处理读取到队列操作
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run([image_batch, label_batch]))

        # 回收线程
        coord.request_stop()
        coord.join(threads)
