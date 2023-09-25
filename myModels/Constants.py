import torch

is_normalization: bool = False
batch_train: bool = False
n_features: int = 28
n_epochs: int = 100
learning_rate: float = 1e-2
weight_decay: float = 0
batch_size: int = 2 ** 10  # 负样本过少，CNN批训练样本少会导致无法训练
rand_dim: int = 1  # 随机数维度
rand_chanel: int = 20  # 随机数通道数
n = 8  # 生成的样本数，可指定
z_shape = [n, rand_chanel, rand_dim]
cnn_out: int = 16  # ！！！cnn模型输出的特征数，尽量设置为偶数。

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ！！！ 训练和验证数据集的路径和文件名，一次导入一个CSV文件，想要操作DataFrame可以在Data_process中的myDataset类
train_data_path = '../Part_Data/'
train_data_name = 'Data_Train_1.csv'
train_data_name_client = 'Data_Train_'
val_data_path = '../Part_Data/'
val_data_name = 'Data_Train_2.csv'

path_train = train_data_path + train_data_name
path_train_client = train_data_path + train_data_name_client
path_val = val_data_path + val_data_name

model_save_path = './models/'
linear_model_name = 'linear_model.pt'
cnn_model_name = 'cnn_model.pt'
gan_model_name = 'gan_model.pt'
