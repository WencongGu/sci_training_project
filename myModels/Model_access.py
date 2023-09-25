import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import datetime
from typing import Any

from Constants import *


class ModelAccess:
    def __init__(self, model, optimizer, loss_fn, train_data, val_data=None, n_epochs=n_epochs,
                 pattern='linear', batch_train=False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.val_data = val_data
        self.n_epochs = n_epochs
        self.batch_train = batch_train
        if batch_train:
            self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        assert pattern in ['linear', 'cnn'], '可选模型：linear，cnn'
        self.pattern = pattern

    def training_loop(self):
        print('开始训练', datetime.datetime.now())
        if self.batch_train:
            print('使用批次训练，批次大小：{}'.format(self.train_loader.batch_size))

        for epoch in range(1, 1 + self.n_epochs):
            loss_train = 0
            self.model.train()
            if self.batch_train:
                for data, label in self.train_loader:

                    if self.pattern == 'linear':
                        data = data.view(-1, n_features)
                        outputs = self.model(data)
                    else:
                        outputs = self.model(data)[0]
                    loss = self.loss_fn(outputs, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_train += loss.item()

                if epoch == 1 or epoch % 10 == 0:
                    print('{}  循环轮次： {}\t\t*loss:  {:.5f}'.format(datetime.datetime.now(), epoch,
                                                                  loss_train / len(self.train_loader)))
            else:
                data = self.train_data.data
                label = self.train_data.label
                if self.pattern == 'linear':
                    data = data.view(-1, n_features)
                    outputs = self.model(data)
                else:
                    outputs = self.model(data)[0]
                loss = self.loss_fn(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()
                if epoch == 1 or epoch % 10 == 0:
                    print('{}  循环轮次： {}\t\t*loss:  {:.5f}'.format(datetime.datetime.now(), epoch, loss_train))

        print('CNN训练完成', datetime.datetime.now())

    def validate(self, val_data=None):
        self.val_data = val_data
        assert self.val_data is not None, "验证集缺失"

        for name, dataset in [('训练集', self.train_data), ('验证集', self.val_data)]:
            correct = 0
            total = 0.1
            self.model.eval()
            with torch.no_grad():
                data = dataset.data
                label = dataset.label
                if self.pattern == 'linear':
                    data = data.view(-1, n_features)
                    outputs = self.model(data)
                if self.pattern == 'cnn':
                    outputs = self.model(data)[0].max(dim=1)
                _, predict = outputs
                total += label.shape[0]
                correct += (predict == label).int().sum()
            print('{}准确率：{:.2f}'.format(name, correct / total))

    def visualization(self):
        w = SummaryWriter(log_dir='./vis/vis_cnn')
        w.add_graph(self.model, torch.rand([144, 1, 28]))
        w.close()

    def save_model(self):
        torch.save(self.model.state_dict(), model_save_path + cnn_model_name)

    def load_model(self):
        with open(model_save_path + cnn_model_name):
            self.model.load_state_dict(torch.load(model_save_path + cnn_model_name))

    def __call__(self, dataset, is_predict=False) -> Any:
        data = dataset.data
        if self.pattern == 'cnn':
            if is_predict:
                return nn.SoftMax(self.model(data)[0], dim=1)
            return self.model(data)[1]
        else:
            return self.model(data)