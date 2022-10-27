import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision  # 数据库模块

# 数据预处理
# 将training data转化成torch能够使用的DataLoader，这样可以方便使用batch进行训练
torch.manual_seed(1)  # reproducible 将随机数生成器的种子设置为固定值，这样，当调用时torch.rand(x)，结果将可重现

# Hyper Parameters
EPOCH = 1  # 训练迭代次数
BATCH_SIZE = 50  # 分块送入训练器
LR = 0.001  # 学习率 learning rate

train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存位置 若没有就新建
    train=True,  # training set
    transform=torchvision.transforms.ToTensor(),  #
    # converts a PIL.Image or numpy.ndarray to torch.FloatTensor(C*H*W) in range(0.0,1.0)
    download=True
)

test_data = torchvision.datasets.MNIST(root='./MNIST/')

train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000] / 255.
# torch.unsqueeze 返回一个新的张量，对输入的既定位置插入维度 1

test_y = test_data.test_lables[:2000]


# 数据预处理


# 定义网络结构
# 1）在Pytorch中激活函数Relu也算是一层layer
# 2）需要·实现·forward()方法，用于网络的前向传播，而反向传播只需要·调用·Variable.backward()即可。
# 输入的四维张量[N, C, H, W]
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # nn.Conv2d 二维卷积 先实例化再使用 在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv2d(in_channels=1,  # input height 必须手动提供 输入张量的channels数
                      out_channels=16,  # n_filter 必须手动提供 输出张量的channels数
                      kernel_size=5,  # filter size 必须手动提供 卷积核的大小
                      # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
                      stride=1,  # filter step 卷积核在图像窗口上每次平移的间隔，即所谓的步长
                      padding=2  # con2d出来的图片大小不变 Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上
                      ),  # output shape (16,28,28) 输出图像尺寸计算公式是唯一的 # O = （I - K + 2P）/ S +1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # 2x2采样，28/2=14，output shape (16,14,14) maxpooling有局部不变性而且可以提取显著特征的同时降低模型的参数，从而降低模型的过拟合
        )
        # 只需要一维卷积
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32,7,7)
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        # 因上述几层网络处理后的output为[32,7,7]的tensor，展开即为7*7*32的一维向量，接上一层全连接层，最终output_size应为10，即识别出来的数字总类别数
        # 在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层 7*7*32, num_classes

    def forward(self, x):
        x = self.conv1(x)  # 卷一次
        # x = self.conv2(x)  # 卷两次
        x = x.view(x.size(0), -1)  # flat (batch_size, 32*7*7)
        # 将前面多维度的tensor展平成一维 x.size(0)指batchsize的值
        # view()函数的功能根reshape类似，用来转换size大小
        # 不需要全连接层
        # output = self.out(x)  # fc out全连接层 分类器
        return x


# 查看网络结构
cnn = CNN()
print(cnn)  # 使用print(cnn)可以看到网络的结构详细信息，可以看到ReLU()也是一层layer

# 训练 需要特别指出的是记得每次反向传播前都要清空上一次的梯度，optimizer.zero_grad()
# optimizer 可以指定程序优化特定的选项，例如学习速率，权重衰减等
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # torch.optim是一个实现了多种优化算法的包

# loss_fun CrossEntropyLoss 交叉熵损失
loss_func = nn.CrossEntropyLoss()  # 该损失函数结合了nn.LogSoftmax()和nn.NLLLoss()两个函数 适用于分类

# training loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        output = cnn(batch_x)  # 输入训练数据
        loss = loss_func(output, batch_y)  # 计算误差 #　实际输出，　期望输出
        optimizer.zero_grad()  # 清空上一次梯度
        loss.backward()  # 误差反向传递 只需要调用.backward()即可
        optimizer.step()  # cnn的优化器参数更新

# 预测结果
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

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
# import sklearn.model_selection
# import torch
# import numpy as np
# import pandas as pd
#
# import data_set
#
# from sklearn.model_selection import train_test_split
#
#
# class dataLoader():
#     data = pd.read_csv(data_set.File_Train)
#     X_train, X_test, y_train, y_test =
#     sklearn.model_selection.train_test_split(data, test_size=0,
#                                              train_size=data.__format__(), random_state=0,
#                                              stratify=y_train)
