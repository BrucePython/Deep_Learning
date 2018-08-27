# 读取是【线程】和【队列】合作的流程
# 主线程执行main，子线程执行def

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def picread(file_list):
    """
    读取狗【图片数据】到【张量】
    :param file_list: 路径+文件名的列表，（是一个1阶张量）
    """
    # 1. 构造文件队列（包括路径+文件名）
    file_queue = tf.train.string_input_producer(file_list)

    # 2. 利用【图片读取器】去【读取】文件队列的内容
    reader = tf.WholeFileReader()       # 构造一个图片读取器实例
    # 默认一次读取一张图片，没有形状
    key, value = reader.read(file_queue)
    # 输出：Tensor("ReaderReadV2:1", shape=(), dtype=string)

    # 3. 对图片数据进行解码; shape：从（）变成 （?,?,?）; dtype: 从string 变成 uint8
    image = tf.image.decode_jpeg(value)
    # 输出：Tensor("ReaderReadV2:1", shape=(), dtype=string)

    # 形状必须固定才能进行批处理，不能用（？，？，？）
    # 4. 图片的形状固定、大小处理(全部统一)，算法训练要求样本的特征数量一样
    # 固定 200*200 ==> [200,200]
    image_resize = tf.image.resize_images(image, [200, 200])
    # 输出：Tensor("batch:0", shape=(200, 200, ?), dtype=float32)

    # 4.1 设置图片形状, 设置成3通道
    image_resize.set_shape([200,200,3])
    # print(image_resize)
    # 输出：Tensor("batch:0", shape=(200, 200, 3), dtype=float32)

    # 5. 进行批处理
    # 包含tensor的列表
    # batch_size:取出的大小
    # capacity：队列的大小
    image_batch = tf.train.batch([image_resize], batch_size=10, num_threads=1,capacity=10)

    return image_batch

if __name__ == '__main__':
    # 第一阶段：构造文件队列
    # 指定文件读取的路径
    file_name = os.listdir("./testA/")   # 返回的仅仅只是文件路径
    # 拼接路径+文件名, 用列表保存
    file_list = [os.path.join("./testA/", file) for file in file_name]
    # print(file_list)

    image_batch = picread(file_list)
    # 输出4D张量：Tensor("batch:0", shape=(10, 200, 200, 3), dtype=float32)

    # 把图片打印出来
    with tf.Session() as sess:

        # 创建线程回收的协调员
        coord = tf.train.Coordinator()

        # 需要手动开启子线程，去进行批处理读取到队列操作
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        print(sess.run(image_batch))

        # 回收线程
        coord.request_stop()
        coord.join(threads)
