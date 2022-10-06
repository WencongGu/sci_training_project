# 目前负责调试各部分是否能正常运行
import numpy as np
import pandas as pd
import time

from _testcapi import INT_MAX

import data_set
import Data_Part
import Data_Upset
import T_C_Part
import XGBoost_train
from Fed_XGBboost import FED_XGB


def _grad(objective, y_hat, Y):
    """
    计算目标函数的一阶导
    支持linear和logistic
    """
    if objective == 'logistic':
        y_hat = 1.0 / (1.0 + np.exp(-y_hat))
        return y_hat - Y
    elif objective == 'linear':
        return y_hat - Y
    else:
        raise KeyError('temporarily: use linear or logistic')


def _hess(objective, y_hat, Y):
    """
    计算目标函数的二阶导
    支持linear和logistic
    """
    if objective == 'logistic':
        y_hat = (1 - 1.0 / (1.0 + np.exp(-y_hat))) / (1. + np.exp(-y_hat))
        return y_hat * (1.0 - y_hat)
    elif objective == 'linear':
        return np.array([1.] * Y.shape[0])
    else:
        raise KeyError('temporarily: use linear or logistic')


if __name__ == '__main__':
    csv_path_p = data_set.File_Upset
    csv_path = data_set.File_Train
    save_dir = 'Part_Data'
    # 执行打乱操作
    Data_Upset.get_label_data()
    # 对数据集进行训练集和测试集的分割
    T_C_Part.Tc_Part().split_csv(csv_path_p)
    # 将训练集平均分配给各用户机
    Data_Part.PyCSV().split_csv(csv_path, save_dir)

    # 客户端训练(每个客户端单独建树)
    # users = data_set.num_user
    # client_gi = []
    # client_hi = []
    # ave_gi = None
    # ave_hi = None
    # # 5轮联合更新
    # for i in range(5):
    #     f1_score = []
    #     roc_auc = []
    #     ave_f1 = 0.00
    #     ave_roc_auc = 0
    #     hi_single_len = 0
    #     gi_single_len = 0
    #     for k in range(users):
    #         client_file = 'Part_Data/Data_Trian_' + str(k + 1) + '.csv'
    #         client_g_h = XGBoost_train.client_train(client_file, ave_gi, ave_hi, i)
    #         # 得到gi和hi以及其余评估值
    #         hi_single_len = client_g_h[1].shape[0]
    #         gi_single_len = client_g_h[0].shape[0]
    #         client_hi.append(client_g_h[1])
    #         client_gi.append(client_g_h[0])
    #         f1_score.append(client_g_h[2])
    #         roc_auc.append(client_g_h[3])
    #         assert type(client_g_h[0]) == pandas.Series
    #         assert type(client_g_h[1]) == pandas.Series
    #         # assert type(client_g_h[2]) == list
    #
    #     # 计算平均的gi和hi
    #     ave_gi = pandas.Series(np.zeros(shape=(gi_single_len,)))
    #     ave_hi = pandas.Series(np.zeros(shape=(hi_single_len,)))
    #     for index, value in enumerate(ave_gi):
    #         for client_gi_value in client_gi:
    #             ave_gi[index] += client_gi_value[index]
    #     for index, value in enumerate(ave_hi):
    #         for client_hi_value in client_hi:
    #             ave_hi[index] += client_hi_value[index]
    #     for j in f1_score:
    #         ave_f1 += j
    #     ave_f1 = ave_f1 / len(f1_score)
    #     # ave_roc_auc += j
    #     print("FED F1_score is: {}".format(ave_f1))
    #     # print(f"AUC Score is: {ave_roc_auc}")
    #     ave_hi = ave_hi / len(client_hi)
    #     ave_gi = ave_gi / len(client_gi)
    #     ave_gi = np.array(ave_gi)
    #     ave_hi = np.array(ave_hi)
    #     print(ave_gi)
    #     print(ave_hi)

    # 客户端训练
    users = data_set.num_user

    # client_g_h = XGBoost_train.server_train(client_file)
    """
    根据训练数据集X和标签集Y训练出树结构和权重
    """
    client_gi = []
    client_hi = []
    ave_gi = None
    ave_hi = None
    gi_single_len = INT_MAX
    hi_single_len = INT_MAX
    for i in range(users):
        client_file = 'Part_Data/Data_Trian_' + str(i + 1) + '.csv'
        df = pd.read_csv(client_file)
        X_train = df[df.columns[:-1].tolist()]
        Y_train = df[df.columns[-1]]
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError('X and Y must have the same length!')
        X = X_train.reset_index(drop=True)
        Y = Y_train.values
        y_hat = np.array([data_set.base_score] * Y.shape[0])
        t0 = time.time()
        gi = _grad('logistic', y_hat, Y)
        hi = _hess('logistic', y_hat, Y)
        assert type(gi) == np.ndarray
        assert type(hi) == np.ndarray
        gi_single_len = min(gi_single_len, gi.shape[0])
        hi_single_len = min(hi_single_len, hi.shape[0])
        client_hi.append(hi)
        client_gi.append(gi)

    # 计算平均的gi和hi
    sum_gi = pd.Series(np.zeros(shape=(gi_single_len,)))
    sum_hi = pd.Series(np.zeros(shape=(hi_single_len,)))
    for index, value in enumerate(sum_gi):
        for client_gi_value in client_gi:
            sum_gi[index] += client_gi_value[index]
    for index, value in enumerate(sum_hi):
        for client_hi_value in client_hi:
            sum_hi[index] += client_hi_value[index]

    sum_gi = np.array(sum_gi)
    sum_hi = np.array(sum_hi)
