# import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# import numpy as np
#
#
# # 数据获取及预处理
# class MNISTLoader():
#     def __init__(self):
#         mnist = tf.keras.datasets.mnist
#         (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
#         # 给数据增加一个通道，也就是一个维度，图片色彩维度
#         self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
#         self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
#         self.train_label = self.train_label.astype(np.float32)
#         self.test_label = self.test_label.astype(np.float32)
#         self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
#
#     def get_batch(self, batch_size):
#         index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
#         return self.train_data[index, :], self.train_label[index]
#
#
# # 模型的构建
# class CNN(tf.keras.Model):
#     def __init__(self):
#         # 关联父类构造函数
#         super(CNN, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(
#             filters=32,  # 卷积层神经元个数
#             kernel_size=[5, 5],  # 感受野大小
#             padding='same',  # 是否进行边界填充，Same填充后输入输出维度一样
#             activation=tf.nn.relu  # 激活函数
#         )
#         self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
#         self.conv2 = tf.keras.layers.Conv2D(
#             filters=64,
#             kernel_size=[5, 5],
#             padding='same',
#             activation=tf.nn.relu
#         )
#         self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
#         self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
#         self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=10)
#
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         output = tf.nn.softmax(x)
#         return output
#
#
# # 模型训练
# if __name__ == '__main__':
#
#     # 定义一些超参数
#     num_epochs = 10  # 迭代次数
#     batch_size = 50  # 每批数据的大小
#     learning_rate = 0.001  # 学习率
#
#     model = CNN()
#     data_loader = MNISTLoader()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#     num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
#     for batch_index in range(num_batches):
#         X, y = data_loader.get_batch(batch_size)
#         with tf.GradientTape() as tape:
#             y_pred = model(X)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
#             loss = tf.reduce_mean(loss)
#
#         # print("epoch %d: loss %f" % (, loss.numpy()))
#         grads = tape.gradient(loss, model.variables)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#
#     # 模型评估
#     sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#     num_batches = int(data_loader.num_test_data // batch_size)
#     for batch_index in range(num_batches):
#         start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
#         y_pred = model.predict(data_loader.test_data[start_index: end_index])
#         sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
#     print("test accuracy: %f" % sparse_categorical_accuracy.result())
import sklearn.model_selection
import torch
import numpy as np
import pandas as pd

import data_set

from sklearn.model_selection import train_test_split


class dataLoader():
    data = pd.read_csv(data_set.File_Train)
    X_train, X_test, y_train, y_test =
    sklearn.model_selection.train_test_split(data, test_size=0,
                                             train_size=data.__format__(), random_state=0,
                                             stratify=y_train)