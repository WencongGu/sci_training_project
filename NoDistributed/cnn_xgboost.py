import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision import transforms
import xgboost as xgb
import matplotlib.pyplot as plt

# 原始数据的维数
input_dim = 28
path = '../Part_Data/Data_Train.csv'
path_test = '../Data_Check/Data_Test.csv'

train_data = pd.read_csv(path)
test_data = pd.read_csv(path_test)
data_pd = train_data.iloc[:, :input_dim]
label_pd = train_data['Class']
data_test_pd = test_data.iloc[:, :input_dim]
label_test_pd = test_data['Class']
# 转换为PyTorch的Tensor格式

data_np = data_pd.to_numpy()
label_np = label_pd.to_numpy()
data_test_np = data_test_pd.to_numpy()
label_test_np = label_test_pd.to_numpy()
train_data = torch.from_numpy(data_np).float()
train_labels = torch.from_numpy(label_np).long()
test_data = torch.from_numpy(data_test_np).float()
test_labels = torch.from_numpy(label_test_np).long()


# 定义CNN模型
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(32 * 13, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, input_dim))
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 创建CNN模型实例
cnn_model = CNNModel()

# 定义训练参数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

# 训练CNN模型
num_epochs = 10
batch_size = 128
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 提取特征
train_features = cnn_model(train_data).detach().numpy()
test_features = cnn_model(test_data).detach().numpy()

# 创建XGBoost模型
xgb_model = xgb.XGBClassifier()

# 训练XGBoost模型
xgb_model.fit(train_features, train_labels)

# 使用XGBoost模型进行预测
predictions = xgb_model.predict(test_features)

# 打印预测结果
print(predictions)
y_test = test_labels
y_pred_proba = predictions

plt.figure(1)
# plt.plot(y_test, c="r", label="y_test")
# plt.plot(y_pred_proba, c="b", label="y_pred")
plt.scatter(np.array(range(y_test.shape[0])), y_test, c='red', label="y_test")
plt.scatter(np.array(range(y_pred_proba.shape[0])), y_pred_proba, c='blue', label="y_pred")
plt.legend()
plt.show()

# 绘制roc曲线
plt.figure(2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

print('roc_auc_score:', roc_auc_score(y_test, y_pred_proba))
