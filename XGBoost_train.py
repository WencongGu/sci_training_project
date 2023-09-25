import xgboost as xgbf
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
from Fed_XGBboost import FED_XGB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import joblib
# from re_FedXGBoost import FED_XGB
import data_set
import numpy
import csv


def calculate_f1(y, t):
    y_result = [1. if i > 0.5 else 0. for i in y]
    print('f1score for client =', f1_score(t, y_result))
    return f1_score(t, y_result)


# def client_train(File_Name, gi, hi, round):
#     df = pd.read_csv(File_Name)
#     X_train = df[df.columns[:-1].tolist()]
#     y_train = df[df.columns[-1]]
#     FML = FED_XGB(learning_rate=0.1,
#                   n_estimators=5  # 总共迭代次数，每进行一轮进行一次全局更新
#                   , max_depth=4, min_child_weight=0.2, gamma=0.03,
#                   objective='logistic')
#     # 得到y_hat
#     client = FML.fit(X_train, y_train, gi, hi, round)
#     client_y_hat = client[0]
#     # test
#     te = pd.read_csv('Data_Check/Data_Test.csv')
#     X_test = te[te.columns[:-1].tolist()]
#     y_test = te[te.columns[-1]]
#     y_pred = FML.predict_raw(X_test)
#     # y_pred_proba = 1./(1.+np.exp(-X_test))
#     y_pred_proba = FML.predict_prob(X_test)
#     f1_pred = calculate_f1(y_pred, y_test)
#     # 返回一阶导和二阶导、f1的值、roc
#     return [client[1], client[2], f1_pred,
#             roc_auc_score(y_test, y_pred_proba)]
#     # FML.save_model('model/clFML-mode.model')

def server_train(all_client, all_client_y):
    FML = FED_XGB(learning_rate=0.1,
                  n_estimators=1  # 总共迭代次数，每进行一轮进行一次全局更新
                  , max_depth=4, min_child_weight=0.2, gamma=0.03,
                  objective='logistic')
    # 得到y_hat
    FML.fit_server(all_client, all_client_y)
    # client_y_hat = client[0]
    # test
    # te = pd.read_csv('Data_Check/Data_Test_cnn.csv')
    te = pd.read_csv('Data_Check/Data_Test.csv')
    X_test = te[te.columns[:-1].tolist()]
    y_test = te[te.columns[-1]]
    y_pred = FML.predict_raw(X_test)
    # y_pred_proba = 1./(1.+np.exp(-X_test))
    y_pred_proba = FML.predict_prob(X_test)
    y_test.to_csv('Data_Check/ytest.csv')
    y_pred.to_csv('Data_Check/ypred.csv')
    print('y_test:', y_test)
    print('y_pred:', y_pred)
    print('y_pred_prob:', y_pred_proba)
    f1_pred = calculate_f1(y_pred, y_test)

    # 保存模型,我们想要导入的是模型本身，所以用“wb”方式写入，即是二进制方式
    # joblib.dump(FML, 'FML_model.dat')

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

    # 返回一阶导和二阶导、f1的值、roc
    return roc_auc_score(y_test, y_pred_proba)
    # FML.save_model('model/clFML-mode.model')
