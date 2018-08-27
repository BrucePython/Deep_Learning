import tensorflow as tf

class Lr():
    def __init__(self):
        self.learning_rate=0.01

    def input(self):
        x_data = tf.random_normal(shape=[100,1],mean=0.0,stddev=1.0,name="x_data")
        y_true = tf.matmul(x_data,[[0.7]]) + 0.8

        return x_data, y_true

    def inference(self,feature):
        """建立模型"""
        with tf.variable_scope("linear_model"):
            self.weight = tf.Variable(tf.random_normal(shape=[1,1],mean=0.0,stddev=1.0),name="weight")
            self.bias = tf.Variable(tf.random_normal(shape=[1],mean=0.0,stddev=1.0),name="bias")
            y_predict = tf.matmul(feature,self.weight)+self.bias

        return y_predict

    def loss(self, y_predict, y_true):
        with tf.variable_scope("losses"):
            loss = tf.reduce_mean(tf.square(y_true - y_predict))
        return loss

    def sgd_op(self,loss):
        with tf.variable_scope("train_op"):
            train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            return train_op

    def merge_summary(self,loss):
        # 收集
        tf.summary.scalar("loss",loss)
        tf.summary.histogram("w",self.weight)
        tf.summary.histogram("b",self.bias)
        # 合并
        merged = tf.summary.merge_all()
        return merged



    def train(self):
        g = tf.get_default_graph()
        with g.as_default():
            x_data, y_true = self.input()
            y_predict = self.inference(x_data)
            loss = self.loss(y_true,y_predict)
            train_op = self.sgd_op(loss)
            merge = self.merge_summary(loss)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                file_writer = tf.summary.FileWriter("./", graph=sess.graph)
                print("模型初始化参数：【权重】为%f, 【偏置】为%f"%(self.weight.eval(),self.bias.eval()))
                for i in range(1000):
                    sess.run(train_op)
                    summary = sess.run(merge)
                    file_writer.add_summary(summary, i)
                    print("第【%d】步，模型优化后参数：【损失】为%f，【权重】为%f, 【偏置】为%f"%(i,loss.eval(), self.weight.eval(),self.bias.eval()))

if __name__ == '__main__':
    lr = Lr()
    lr.train()