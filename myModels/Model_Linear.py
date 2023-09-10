import torch.nn as nn
import torch.optim as optim

from Constants import *
from Data_process import data_train, data_val
from Model_access import ModelAccess

model_linear = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.Tanh(),
        nn.Linear(64, 2),
        nn.LogSoftmax(dim=1))
# nn.Tanh(),
# nn.Linear(32,2))
loss_fn_linear = nn.NLLLoss()  # 使用LogSoftMax则不用交叉熵损失
optimizer_linear = optim.Adam(model_linear.parameters(), lr=learning_rate, weight_decay=weight_decay)
# acc_linear=ModelAccess(model=model_linear,optimizer=optimizer_linear,loss_fn=loss_fn_linear,train_data=data_train,n_epochs=n_epochs,pattern='linear')
