import datetime
import time

import torch.nn as nn
from torch.utils.data import DataLoader

from Constants import *
from Data_process import data_train, data_val


class Generator(nn.Module):
    """
    输入：[样本数, 通道数(20), 1]
    输出：[样本数, 通道数(1), 特征数(28)]
    输入：[n, 20, 1]
    输出：[n, 1, 28]
    """

    # 输入1维随机初始值经过生成器后变为28维，即假样本
    def __init__(self, z_dim=rand_chanel, data_size=32) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.ConvTranspose1d(z_dim, data_size * 8, kernel_size=4),
                nn.BatchNorm1d(data_size * 8),
                nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
                nn.ConvTranspose1d(data_size * 8, data_size * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(data_size * 4),
                nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
                nn.ConvTranspose1d(data_size * 4, data_size * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(data_size * 2),
                nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
                nn.ConvTranspose1d(data_size * 2, data_size, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm1d(data_size),
                nn.ReLU(inplace=True))

        self.last = nn.Sequential(
                nn.ConvTranspose1d(data_size, 1, kernel_size=4, stride=2, padding=1),
                nn.Tanh())

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    """
    输入：[样本数, 通道数(1), 特征数(1)]
    输出：[样本数, 通道数(1), 1], features
    输入：[n, 1, 28]
    输出：[n, 1, 1], features
    """

    def __init__(self, data_size=32) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
                nn.Conv1d(1, data_size, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
                nn.Conv1d(data_size, data_size * 2, kernel_size=3, stride=1, padding=2),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
                nn.Conv1d(data_size * 2, data_size * 4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
                nn.Conv1d(data_size * 4, data_size * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True))

        self.last = nn.Conv1d(data_size * 8, 1, kernel_size=4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        feature = out.view(out.size()[0], -1)

        out = self.last(out)

        return out, feature


class ModelAccess_AnoGAN:
    def __init__(self, G=Generator(), D=Discriminator(), data_train=None, data_val=None, n_epochs=n_epochs,
                 batch_train=False):
        self.G = G
        self.D = D
        self.data_train = data_train.data_pos
        self.data_val = data_val
        self.g_lr, self.d_lr = 0.0001, 0.0004
        self.beta1, self.beta2 = 0.0, 0.9
        self.G.apply(self.weight_init)
        self.D.apply(self.weight_init)
        # 优化器参数
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2))
        # BCEWithLogitsLoss是先将输入数据乘以Logistic，再计算二进制交叉熵
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        self.n_epochs = n_epochs
        self.batch_train = batch_train

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # Conv1d和ConvTranspose1d的初始化
            nn.init.normal_(m.weight.data, 0., 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            # BatchNorm1d的初始化
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_model(self, data_train=None):
        if data_train is None:
            data_train = self.data_train
        else:
            data_train=data_train.data_pos

        # 计算开销较大，能用GPU就用GPU；如果网络相对稳定，则开启加速
        print("使用设备：", device)
        self.G.to(device)
        self.D.to(device)
        self.G.train()
        self.D.train()
        torch.backends.cudnn.benchmark = True

        print('训练开始，时间：{}'.format(datetime.datetime.now()))
        if self.batch_train:
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
            for epoch in range(1, 1 + self.n_epochs):
                t_epoch_start = time.time()
                epoch_g_loss = 0.
                epoch_d_loss = 0.

                print('-------------')
                print('Epoch {}/{}'.format(epoch, self.n_epochs))
                print('-------------')
                for data in train_loader:
                    # 判别器D
                    if data.size()[0] == 1:  # 小批次尺寸不能为1，会导致批次归一化失败
                        continue

                    # 如果能使用GPU，则将数据送入GPU中
                    data = data.to(device)

                    # 创建正确答案标签和伪造数据标签
                    mini_batch_size = data.size()[0]
                    label_real = torch.full((mini_batch_size,), 1.).to(device)
                    label_fake = torch.full((mini_batch_size,), 0.).to(device)

                    d_out_real, _ = self.D(data)

                    z_shape = [mini_batch_size, rand_chanel, 1]
                    input_z = torch.randn(z_shape).to(device)
                    fake_images = self.G(input_z)
                    d_out_fake, _ = self.D(fake_images)

                    d_loss_real = self.criterion(d_out_real.view(-1), label_real)
                    d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
                    d_loss = d_loss_real + d_loss_fake

                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # 生成器G
                    input_z = torch.randn(z_shape).to(device)
                    fake_images = self.G(input_z)
                    d_out_fake, _ = self.D(fake_images)

                    g_loss = self.criterion(d_out_fake.view(-1), label_real)

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    epoch_d_loss += d_loss.item()
                    epoch_g_loss += g_loss.item()

                t_epoch_finish = time.time()
                print('循环计次： {} || 本轮D_Loss:{:.4f} \t G_Loss:{:.4f}'.format(
                        epoch, epoch_d_loss / train_loader.batch_size, epoch_g_loss / train_loader.batch_size))
                print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        else:
            for epoch in range(1, 1 + self.n_epochs):
                t_epoch_start = time.time()

                print('-------------')
                print('Epoch {}/{}'.format(epoch, self.n_epochs))
                print('-------------')

                # 判别器D
                data = data_train.to(device)
                mini_batch_size = data.size()[0]
                label_real = torch.full((mini_batch_size,), 1.).to(device)
                label_fake = torch.full((mini_batch_size,), 0.).to(device)
                d_out_real, _ = self.D(data)

                z_shape = [mini_batch_size, rand_chanel, 1]
                input_z = torch.randn(z_shape).to(device)
                fake_images = self.G(input_z)
                d_out_fake, _ = self.D(fake_images)

                d_loss_real = self.criterion(d_out_real.view(-1), label_real)
                d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
                d_loss = d_loss_real + d_loss_fake

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # 生成器G
                input_z = torch.randn(z_shape).to(device)
                fake_images = self.G(input_z)
                d_out_fake, _ = self.D(fake_images)

                g_loss = self.criterion(d_out_fake.view(-1), label_real)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                epoch_d_loss = d_loss.item()
                epoch_g_loss = g_loss.item()

                t_epoch_finish = time.time()
                print('循环计次： {} || 本轮D_Loss:{:.4f} \t G_Loss:{:.4f}'.format(
                        epoch, epoch_d_loss, epoch_g_loss))
                print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        print('训练结束，时间：{}'.format(datetime.datetime.now()))

    def Anomaly_score(self, x, fake_data, D=None, Lambda=0.1):
        """需要无标签的数据，但是可以在内部处理。为方便直接使用my Dataset，先提取其data属性"""
        x = x.data
        if D is None:
            D = self.D

        # 求各条数据各个维度之差的绝对值，并对每个批求和
        residual_loss = torch.abs(x - fake_data)
        residual_loss = residual_loss.view(residual_loss.size()[0], -1)
        residual_loss = torch.sum(residual_loss, dim=1)

        # 将异常样本x和生成样本fake_data输入到识别器D，取出特征量
        _, x_feature = D(x)
        _, G_feature = D(fake_data)

        # 求测试样本x和生成样本fake_data的特征量之差的绝对值，对每个批次求和
        discrimination_loss = torch.abs(x_feature - G_feature)
        discrimination_loss = discrimination_loss.view(
                discrimination_loss.size()[0], -1)
        discrimination_loss = torch.sum(discrimination_loss, dim=1)

        # 将两种损失对每个批次进行加法运算
        loss_each = (1 - Lambda) * residual_loss + Lambda * discrimination_loss

        # 求所有批次的全部损失
        total_loss = torch.sum(loss_each)

        return total_loss, loss_each, residual_loss

    def z_find(self, data_val=None, G=None, D=None, n_epochs=n_epochs):
        """data_val是不带标签的myDataset，应当用myDataset.data属性，不过已经在Anomaly_score()和中处理过了，直接传入myDataset就可以"""
        if G is None:
            G = self.G
        if data_val is None:
            data_val = self.data_val
        n = len(data_val)
        z_shape = [n, rand_chanel, 1]
        z = torch.rand(z_shape, requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=1e-3)
        print('求z。{}'.format(datetime.datetime.now()))
        for epoch in range(1, n_epochs + 1):
            loss, _, _ = self.Anomaly_score(data_val, G(z), D, Lambda=0.1)

            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()

            if epoch % 2 == 0:
                print('epoch {} || finding z, loss_total:{:.0f} '.format(epoch, loss.item()))
        print('完成。{}'.format(datetime.datetime.now()))
        return z

    def validate(self, data_val=None, G=None, D=None, threshold=100, Lambda=0.1, found=None):
        """
        data_val是不带标签的myDataset，应当用myDataset.data属性，不过已经过处理，直接传入myDataset即可
        输出numpy数组，0代表原数据中相同位置为正常样本，1代表异常样本
        """
        if data_val is None:
            data_val = self.data_val
        if G is None:
            G = self.G
        if D is None:
            D = self.D
        if found is not None:
            if found.size()[0] != len(data_val):
                raise '注意，传入的found的size不正确，请检查'
            z = found
        else:
            z = self.z_find(data_val)

        self.G.eval()
        self.D.eval()
        fake_data = G(z)
        _, loss_each, _ = self.Anomaly_score(
                data_val, fake_data, D, Lambda=Lambda)
        loss_each = loss_each.cpu().detach().numpy()
        # predict=[ 'normal'*int(i<threshold)+'abnormal'*int(i>=threshold) for i in loss_each]
        predict = (loss_each <= threshold)
        acc = (predict == data_val.label.numpy()).sum() / len(data_val)
        print('准确率：{}'.format(acc))
        return predict

    def __call__(self, data_val=None, G=None, D=None, threshold=100, Lambda=0.1, found=None):
        pred = self.validate(data_val=data_val, G=G, D=D, threshold=threshold, Lambda=Lambda, found=found)
        return pred

#
# acc = ModelAccess_AnoGAN(data_train=data_train, data_val=data_val, n_epochs=1)
# print(a:=acc.D(data_val.data_pos[:10])[0])
# acc.train_model(data_val)
# print(b:=acc.D(data_val.data_pos[:10])[0])
# p = acc.validate()
# print(len(data_train), p)
# print(a==b)
