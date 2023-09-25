# 导入需要的库
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import numpy as np

# client_file = '../Part_Data/Data_Train.csv'  # 不加cnn
client_file = '../myModels/data_transformed/tensor_all.csv'  # 加cnn
# client_file_cnn = 'myModels/data_transformed/tensor_' + str(i + 1) + '.csv'
df = pd.read_csv(client_file)
# df = pd.read_csv(client_file)
X_train = df[df.columns[:-1].tolist()]
y_train = df[df.columns[-1]]
# te = pd.read_csv('../Data_Check/Data_Test.csv')  # 不加cnn
te = pd.read_csv('../Data_Check/Data_test_cnn.csv')  # 加cnn
X_test = te[te.columns[:-1].tolist()]
y_test = te[te.columns[-1]]
label_mapping = {0: 'V1', 1: 'V2', 2: 'V3', 3: 'V4', 4: 'V5', 5: 'V6', 6: 'V7', 7: 'V8',
                 8: 'V9', 9: 'V10', 10: 'V11', 11: 'V12', 12: 'V13', 13: 'V14', 14: 'V15',
                 15: 'V16', 16: 'V17', 17: 'V18', 18: 'V19', 19: 'V20', 20: 'V21', 21: 'V22',
                 22: 'V23', 23: 'V24', 24: 'V25', 25: 'V26', 26: 'V27', 27: 'V28', 28: 'Amount'}
# 将数据分为训练数据和测试数据
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)
# 训练XGBoost分类器
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# xgb.plot_tree(model)
# 使用测试数据预测类别
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

# plt.plot(y_test, c="r", label="y_test")
# plt.plot(y_pred_proba, c="b", label="y_pred")
plt.figure(1)
plt.figure(figsize=(20, 5))  # 6，8分别对应宽和高
plt.scatter(np.array(range(y_test.shape[0])), y_test, c='red', label="y_test")
plt.scatter(np.array(range(y_pred.shape[0])), y_pred, c='blue', label="y_pred")
plt.legend()
plt.show()

plt.figure(2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

print('roc_auc_score:', roc_auc_score(y_test, y_pred))
