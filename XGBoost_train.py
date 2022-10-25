import xgboost as xgbf
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
from Fed_XGBboost import FED_XGB
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
                  n_estimators=5  # 总共迭代次数，每进行一轮进行一次全局更新
                  , max_depth=4, min_child_weight=0.2, gamma=0.03,
                  objective='logistic')
    # 得到y_hat
    FML.fit_server(all_client, all_client_y)
    # client_y_hat = client[0]
    # test
    te = pd.read_csv('Data_Check/Data_Test.csv')
    X_test = te[te.columns[:-1].tolist()]
    y_test = te[te.columns[-1]]
    y_pred = FML.predict_raw(X_test)
    # y_pred_proba = 1./(1.+np.exp(-X_test))
    y_pred_proba = FML.predict_prob(X_test)
    f1_pred = calculate_f1(y_pred, y_test)
    # 返回一阶导和二阶导、f1的值、roc
    return roc_auc_score(y_test, y_pred_proba)
    # FML.save_model('model/clFML-mode.model')
