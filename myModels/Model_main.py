import numpy as np
import pandas as pd

from Constants import *
from Data_process import MyDataset, data_train, data_val
from Model_CNN import model_cnn, loss_fn_cnn, optimizer_cnn
from Model_Linear import model_linear, loss_fn_linear, optimizer_linear
from Model_access import ModelAccess
from Model_AnoGAN import ModelAccess_AnoGAN
from Model_EfficientGAN import ModelAccess_EGAN

# 更多参数可以在Constants.py中设置。

# ！！！经过修改，创建数据(MyDataset)时即可以传入csv格式的数据（csv_data参数），也可以传入文件路径（path参数）
# 如：
data = MyDataset(path_train)
# 或者：
data = pd.read_csv(path_train)  # 但是要注意这个数据应当包含标签值
data = MyDataset(csv_data=data)


# 也可以使用在Data_process.py中直接定义好的data_train，data_val，上面已经导入了

# ！！！现在不论是训练模型还是使用模型，传入数据都只需要是MyDataset类型就行，不必多虑
def myLinear():
    acc_linear = ModelAccess(model=model_linear, optimizer=optimizer_linear, loss_fn=loss_fn_linear,
                             train_data=data_train, n_epochs=n_epochs, pattern='linear')
    acc_linear.training_loop()
    acc_linear.validate(data_val)
    acc_linear(data_val)
    return acc_linear


def myCNN():
    acc_cnn = ModelAccess(model=model_cnn, optimizer=optimizer_cnn, loss_fn=loss_fn_cnn, train_data=data_train,
                          n_epochs=5, pattern='cnn')
    acc_cnn.training_loop()
    acc_cnn.validate(data_val)  # 输出验证结果、准确率
    acc_cnn(data_val)  # 调用模型，直接输出转化后的数据
    return acc_cnn

def myAnoGAN():
    acc_anogan=ModelAccess_AnoGAN(data_train=data_train, data_val=data_val)
    acc_anogan.train_model()
    acc_anogan.validate()


def myEGAN():
    acc_egan=ModelAccess_EGAN(data_train=data_train, data_val=data_val)
    acc_egan.train_model()
    acc_egan.validate()


def toCSV(data: torch.Tensor, path='./data_transformed/'):
    data_np = data.view(data.shape[0], -1).detach().numpy()
    np.savetxt(path + 'tensor.csv', data_np, delimiter=',')


acc_cnn = myCNN()

# 示例
n = torch.rand(12000, 1, 28)
print(model_cnn(n))
print(acc_cnn(n))
print(acc_cnn(n) == model_cnn(n))
toCSV(acc_cnn(n))
