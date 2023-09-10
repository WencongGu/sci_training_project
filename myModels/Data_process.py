from torch.utils.data.dataset import Dataset
from torchvision import transforms

import pandas as pd

from Constants import *


class MyDataset(Dataset):
    def __init__(self, path=None, csv_data=None, is_normalization=is_normalization):

        if path is not None:
            # ！！！ 想要对导入数据进行处理在下面这三行代码中处理，完全当成pandas处理就行
            self.train_data = pd.read_csv(path)
            self.data_pd = self.train_data.iloc[:, :n_features]
            self.label_pd = self.train_data['Class']
        elif csv_data is not None:
            self.train_data = csv_data
            self.data_pd = self.train_data.iloc[:, :n_features]
            self.label_pd = self.train_data['Class']
        else:
            raise ValueError('需要传入数据(csv格式，csv_data参数)或数据路径(path参数)')

        self.data_np = self.data_pd.to_numpy()
        self.label_np = self.label_pd.to_numpy()
        self.is_normalization = is_normalization

        self.data = torch.from_numpy(self.data_np)
        self.label = torch.from_numpy(self.label_np)
        if device == 'cuda:0':
            self.data = self.data.cuda()
            self.label = self.label.cuda()
        else:
            self.data = self.data.to(device)
            self.label = self.label.to(device)
        self.data = self.data.float().unsqueeze(dim=1)
        self.data_pos = self.data[self.label == 0]
        self.data_neg = self.data[self.label == 1]

        if self.is_normalization:
            self.data_mean = self.data.view(len(self.data), -1).mean(dim=1)
            self.data_std = self.data.view(len(self.data), -1).mean(dim=1)
            self.normalizer = transforms.Normalize(mean=self.data_mean, std=self.data_std)
            self.data = self.normalizer(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[
            index]  # 想重构代码就应该不要这个label，造成很多麻烦，至于loader，可以使用Mydataset的data和label属性共同生成loader
        # return super().__getitem__(index)

    def __len__(self):
        return self.data.shape[0]


data_train = MyDataset(path=path_train)
data_val = MyDataset(path=path_val)
