import torch.nn as nn
import torch.optim as optim

from Constants import *
from Model_access import ModelAccess
from Data_process import data_train, data_val


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv1d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)  # <1>
        self.batch_norm = nn.BatchNorm1d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')  # <2>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


class NetResDeep(nn.Module):
    def __init__(self, n_chans1=16, n_blocks=5, n_out=cnn_out):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv1d(1, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
                *(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(n_features * self.n_chans1, 32)
        self.fc2 = nn.Linear(32, n_out)
        self.fc3 = nn.Linear(n_out, 2)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.resblocks(out)
        out = out.view(-1, n_features * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        feature = out.detach()
        out = self.fc3(out)
        return out, feature


model_cnn = NetResDeep()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=1e-3, weight_decay=weight_decay)
loss_fn_cnn = nn.CrossEntropyLoss()
# acc_cnn=ModelAccess(model=model_cnn,optimizer=optimizer_cnn,loss_fn=loss_fn_cnn,train_data=data_train,val_data=data_val,n_epochs=5,pattern='cnn')
