# 用于将数据进行CNN网络连接

import pandas
import torch.nn as nn
import torch.nn.functional as fun
import torch.utils.data as data


class CNN(nn.Module):
    def __init__(self, feature_size):
        super(CNN, self).__init__()

        # 定义一维卷积层
        self.conv1d_1 = nn.Conv1d(in_channels=feature_size, out_channels=10, kernel_size=3)
        self.conv1d_2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv1d_3 = nn.Conv1d(in_channels=20, out_channels=30, kernel_size=3)

        # 池化
        self.pooling = nn.MaxPool1d(2)
        # self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # batch_size = x.size(0)
        x = fun.relu(self.pooling(self.conv1d_1(x)))
        x = fun.relu(self.pooling(self.conv1d_2(x)))
        x = fun.relu(self.pooling(self.conv1d_3(x)))
        # x = x.view(batch_size, -1)
        # x = self.fc(x)
        return x


if __name__ == "__main__":
    batch_size = 29
    cnn = CNN(batch_size)
    print(cnn)
    # 引入数据集，发现问题：引入数据集参数错误
    train_set = pandas.read_csv('Part_Data/Data_Trian_1.csv', sep=',', header=0)
    # print(train_set)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    # print(train_loader)
    print(cnn.forward(train_loader))
    # cnn.forward(train_loader)
