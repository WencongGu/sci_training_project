import datetime
import time

import torch.nn as nn
from torch.utils.data import DataLoader

from Constants import *
from Data_process import data_train, data_val


class Generator(nn.Module):
    """
    输入仅要求第0维是样本数
    输出：[样本数，通道数(1)，特征数(28)]
    """

    def __init__(self, z_dim=20):
        super(Generator, self).__init__()

        self.layer1_linear = nn.Sequential(
                nn.Linear(z_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True))

        self.layer2_linear = nn.Sequential(
                nn.Linear(1024, 7 * 128),
                nn.BatchNorm1d(7 * 128),
                nn.ReLU(inplace=True))

        self.layer3_convtrans = nn.Sequential(
                nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True))

        self.last = nn.Sequential(
                nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
                nn.Tanh())

    def forward(self, z):
        z = z.view(z.shape[0], -1)
        out = self.layer1_linear(z)
        out = self.layer2_linear(out)

        out = out.view(z.shape[0], 128, 7)
        out = self.layer3_convtrans(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):
    """
    输入：监督数据[样本数，通道数(1)，特征数(28)], 生成值[样本数，通道数(20)]
    输出：[样本数，1], feature
    """

    def __init__(self, z_dim=20):
        """x是监督数据，z是随机生成数"""
        super(Discriminator, self).__init__()

        self.x_layer1 = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=4,
                          stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))
        # 特征数缩小为1/2，14

        self.x_layer2 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=4,
                          stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1, inplace=True))
        # 特征数缩小为1/2，7

        self.z_layer1 = nn.Linear(z_dim, 512)

        self.last1 = nn.Sequential(
                nn.Linear(64 * 7 + 512, 1024),
                nn.LeakyReLU(0.1, inplace=True))

        self.last2 = nn.Linear(1024, 1)

    def forward(self, x, z):
        """x是监督数据，z是噪声"""

        x_out = self.x_layer1(x)
        x_out = self.x_layer2(x_out)

        z = z.view(z.shape[0], -1)
        z_out = self.z_layer1(z)

        x_out = x_out.view(-1, 64 * 7)
        out = torch.cat([x_out, z_out], dim=1)
        out = self.last1(out)

        feature = out.view(out.size()[0], -1)
        out = self.last2(out)

        return out, feature


class Encoder(nn.Module):
    """
    输入：[样本数，通道数(1)，特征数(28)]
    输出：[样本数，通道数(20)]
    """

    def __init__(self, z_dim=20):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3,
                          stride=1),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3,
                          stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3,
                          stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1, inplace=True))
        # 到这里为止，数据的尺寸为7×1，128个通道

        self.last = nn.Linear(128 * 7, z_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(-1, 128 * 7)
        out = self.last(out)

        return out


class ModelAccess_EGAN:
    def __init__(self, G=Generator(), D=Discriminator(), E=Encoder(), data_train=None, data_val=None, n_epochs=n_epochs,
                 batch_train=False):
        self.G = G
        self.D = D
        self.E = E
        self.data_train = data_train.data_pos
        self.data_val = data_val
        self.lr_ge = 0.0001
        self.lr_d = 0.0001 / 4
        self.beta1, self.beta2 = 0.5, 0.999
        self.G.apply(self.weight_init)
        self.E.apply(self.weight_init)
        self.D.apply(self.weight_init)

        # 优化器参数
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr_ge, betas=(self.beta1, self.beta2))
        self.e_optimizer = torch.optim.Adam(self.E.parameters(), lr=self.lr_ge, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))

        # BCEWithLogitsLoss是先将输入数据乘以Logistic，再计算二进制交叉熵
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        self.n_epochs = n_epochs
        self.batch_train = batch_train

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # Conv2d和ConvTranspose2d的初始化
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            # BatchNorm2d的初始化
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            # 全连接层Linear的初始化
            m.bias.data.fill_(0)

    def train_model(self, data_train=None):
        if data_train is None:
            data_train = self.data_train
        else:
            data_train = data_train.data_pos

        # 计算开销较大，能用GPU就用GPU；如果网络相对稳定，则开启加速
        print("使用设备：", device)
        self.G.to(device)
        self.E.to(device)
        self.D.to(device)
        self.G.train()
        self.E.train()
        self.D.train()
        torch.backends.cudnn.benchmark = True

        print('训练开始，时间：{}'.format(datetime.datetime.now()))
        if self.batch_train:
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
            for epoch in range(1, 1 + self.n_epochs):

                t_epoch_start = time.time()
                epoch_g_loss = 0.
                epoch_e_loss = 0.
                epoch_d_loss = 0.

                print('-------------')
                print('Epoch {}/{}'.format(epoch, self.n_epochs))
                print('-------------')

                for data in train_loader:

                    # 如果小批次的尺寸设置为1，会导致批次归一化处理产生错误，因此需要避免
                    if data.size()[0] == 1:
                        continue

                    mini_batch_size = data.size()[0]
                    label_real = torch.full((mini_batch_size,), 1.).to(device)
                    label_fake = torch.full((mini_batch_size,), 0.).to(device)

                    data = data.to(device)

                    # 判别器D的学习
                    z_out_real = self.E(data)
                    d_out_real, _ = self.D(data, z_out_real)
                    input_z = torch.randn(mini_batch_size, rand_chanel).to(device)
                    fake_data = self.G(input_z)
                    d_out_fake, _ = self.D(fake_data, input_z)
                    d_loss_real = self.criterion(d_out_real.view(-1), label_real)
                    d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
                    d_loss = d_loss_real + d_loss_fake

                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # 生成器G的学习
                    input_z = torch.randn(mini_batch_size, rand_chanel).to(device)
                    fake_images = self.G(input_z)
                    d_out_fake, _ = self.D(fake_images, input_z)
                    g_loss = self.criterion(d_out_fake.view(-1), label_real)

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # 编码器E的学习
                    z_out_real = self.E(data)
                    d_out_real, _ = self.D(data, z_out_real)
                    e_loss = self.criterion(d_out_real.view(-1), label_fake)

                    self.e_optimizer.zero_grad()
                    e_loss.backward()
                    self.e_optimizer.step()

                    epoch_d_loss += d_loss.item()
                    epoch_g_loss += g_loss.item()
                    epoch_e_loss += e_loss.item()

                t_epoch_finish = time.time()
                print('-------------')
                print('循环计次： {} || 本次循环 D_Loss:{:.4f} ||G_Loss:{:.4f} ||E_Loss:{:.4f}'.format(
                        epoch, epoch_d_loss / train_loader.batch_size, epoch_g_loss / train_loader.batch_size,
                               epoch_e_loss / train_loader.batch_size))
                print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        else:
            for epoch in range(1, 1 + self.n_epochs):
                t_epoch_start = time.time()
                print('-------------')
                print('Epoch {}/{}'.format(epoch, self.n_epochs))
                print('-------------')

                data = data_train.to(device)
                mini_batch_size = data.size()[0]
                label_real = torch.full((mini_batch_size,), 1.).to(device)
                label_fake = torch.full((mini_batch_size,), 0.).to(device)

                # 判别器D的学习
                z_out_real = self.E(data)
                d_out_real, _ = self.D(data, z_out_real)
                input_z = torch.randn(mini_batch_size, rand_chanel).to(device)
                fake_data = self.G(input_z)
                d_out_fake, _ = self.D(fake_data, input_z)
                d_loss_real = self.criterion(d_out_real.view(-1), label_real)
                d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
                d_loss = d_loss_real + d_loss_fake

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # 生成器G的学习
                input_z = torch.randn(mini_batch_size, rand_chanel).to(device)
                fake_images = self.G(input_z)
                d_out_fake, _ = self.D(fake_images, input_z)
                g_loss = self.criterion(d_out_fake.view(-1), label_real)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # 编码器E的学习
                z_out_real = self.E(data)
                d_out_real, _ = self.D(data, z_out_real)
                e_loss = self.criterion(d_out_real.view(-1), label_fake)

                self.e_optimizer.zero_grad()
                e_loss.backward()
                self.e_optimizer.step()

                epoch_d_loss = d_loss.item()
                epoch_g_loss = g_loss.item()
                epoch_e_loss = e_loss.item()

                t_epoch_finish = time.time()
                print('-------------')
                print('循环计次： {} || 本次循环 D_Loss:{:.4f} ||G_Loss:{:.4f} ||E_Loss:{:.4f}'.format(
                        epoch, epoch_d_loss, epoch_g_loss, epoch_e_loss))
                print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        print("训练结束，时间：{}".format(datetime.datetime.now()))

    def Anomaly_score(self, x, fake_data, z_out_real, D=None, Lambda=0.1):
        """需要无标签的数据，但是可以在内部处理。为方便直接使用my Dataset，先提取其data属性"""
        x = x.data
        if D is None:
            D = self.D

        # 求各条数据各个维度之差的绝对值，并对每个批求和
        residual_loss = torch.abs(x - fake_data)
        residual_loss = residual_loss.view(residual_loss.size()[0], -1)
        residual_loss = torch.sum(residual_loss, dim=1)

        # 将异常样本x和生成样本fake_data输入到识别器D，取出特征量
        _, x_feature = D(x, z_out_real)
        _, G_feature = D(fake_data, z_out_real)

        # 求测试样本x和生成样本fake_data的特征量之差的绝对值，对每个批次求和
        discrimination_loss = torch.abs(x_feature - G_feature)
        discrimination_loss = discrimination_loss.view(
                discrimination_loss.size()[0], -1)
        discrimination_loss = torch.sum(discrimination_loss, dim=1)

        # 将每个小批次中的两种损失相加
        loss_each = (1 - Lambda) * residual_loss + Lambda * discrimination_loss

        # 对所有批次中的损失进行计算
        total_loss = torch.sum(loss_each)

        return total_loss, loss_each, residual_loss

    def validate(self, data_val=None, G=None, D=None, E=None, threshold=100, Lambda=0.1):
        """
        data_val是不带标签的myDataset，应当用myDataset.data属性，不过已经处理，直接传入myDataset
        输出numpy数组，0代表原数据中相同位置为正常样本，1代表异常样本
        """

        if data_val is None:
            data_val = self.data_val
        if G is None:
            G = self.G
        if D is None:
            D = self.D
        if E is None:
            E = self.E
        z_out_real = E(data_val.data)
        data_reconstract = G(z_out_real)
        _, loss_each, _ = self.Anomaly_score(
                data_val, data_reconstract, z_out_real, D, Lambda=Lambda)

        loss_each = loss_each.cpu().detach().numpy()
        # predict = ['normal' * int(i < 100) + 'abnormal' * int(i >= 100) for i in loss_each]
        predict = (loss_each <= threshold)
        acc = (predict == data_val.label.numpy()).sum() / len(data_val)
        print('准确率：{}'.format(acc))

        return predict

    def __call__(self, data_val=None, G=None, D=None, E=None, threshold=100, Lambda=0.1, found=None):
        pred = self.validate(data_val=data_val, G=G, D=D, E=E, threshold=threshold, Lambda=Lambda, found=found)
        return pred


# acc = ModelAccess_EGAN(data_train=data_train, data_val=data_val, n_epochs=5)
# acc.train_model(data_val)
# p = acc.validate()
