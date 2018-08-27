import tensorflow as tf
import os


class Cifar_raw(object):
    def __init__(self):
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_byte = 1
        self.image_bytes = self.height * self.width * self.channel
        self.all_bytes = self.label_byte + self.image_bytes

    def cifer_reader(self, file_list):
        file_queue = tf.train.string_input_producer(file_list)
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        _, value = reader.read(file_queue)
        image_label_image = tf.decode_raw(value, tf.uint8)  # (?,),    3073个字节
        print(image_label_image)
        label = tf.slice(image_label_image, [0], [self.label_byte])
        label = tf.cast(label, tf.int32)
        image = tf.slice(image_label_image, [self.label_byte], [self.image_bytes])
        image_depth_major = tf.reshape(image, [self.channel, self.height, self.width])
        print(image_depth_major)
        image = tf.transpose(image_depth_major, [1, 2, 0])
        batch_label, batch_image = tf.train.batch([label, image], batch_size=10, num_threads=1, capacity=10)
        return batch_label, batch_image

    def write_to_TFRecordWriter(self, batch_label, batch_image):
        writer = tf.python_io.TFRecordWriter("./cifar.tfrecords")
        for i in range(10):
            # tensor转成Byte
            image = batch_image[i].eval().tostring()
            # tensor转成int64
            label = batch_label[i].eval()[0]

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            # 写入第一样本的example
            writer.write(example.SerializeToString())

        writer.close()  # 关闭文件
        return None

    def read_TFRecord(self):
        print(1)
        file_queue = tf.train.string_input_producer(['./cifar.tfrecords'])
        print(file_queue)
        reader = tf.TFRecordReader()
        _, value = reader.read(file_queue)
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(feature["image"], tf.uint8)
        label = tf.cast(feature["label"], tf.int32)
        image_reshape = tf.reshape(image, [32, 32, 3])
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch


if __name__ == '__main__':
    # file_name = os.listdir("./cifar10/")
    # file_list = [os.path.join("./cifar10/", file) for file in file_name if file[-3:] == "bin"]
    cr = Cifar_raw()
    # batch_label, batch_image = cr.cifer_reader(file_list)
    image_batch, label_batch = cr.read_TFRecord()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(sess.run([image_batch, label_batch]))
        # cr.write_to_TFRecordWriter(batch_label, batch_image)
        coord.request_stop()
        coord.join(threads)
