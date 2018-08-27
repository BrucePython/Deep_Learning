import os
import tensorflow as tf

def read_pic(file_list):
    # 1. 构造队列，准备路径+文件名
    file_queue = tf.train.string_input_producer(file_list)

    # 2.实例化图片读取器，并解码
    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)
    image = tf.image.decode_jpeg(value)

    # 固定成（3，3，3）
    image_resize = tf.image.resize_images(image, [200,200],)    # 每个样本的特征值固定：固定所有图片的长和宽
    image_resize.set_shape([200,200,3])

    # 批处理：
    image_batch = tf.train.batch([image_resize],batch_size=10, num_threads=1, capacity=10)

    return image_batch


if __name__ == '__main__':
    files = os.listdir("./testA")
    file_list = [os.path.join("./testA",file) for file in files]
    image_batch = read_pic(file_list)
    with tf.Session() as sess:

        # 创建线程回收的协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        print(sess.run(image_batch))

        # 回收线程
        coord.request_stop()
        coord.join(threads)