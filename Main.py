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


# def xgb_cart_tree_server(X, depth=0):
#     """
#         递归构造XCart树
#         """
#     if depth > 5:
#         return None
#     best_var, best_cut = None, None
#     max_gain = 0
#     G_left_best, G_right_best, H_left_best, H_right_best = 0, 0, 0, 0
#     client0 = X[0]
#     for item in [x for x in client0.columns if x not in ['g', 'h', 'Class']]:
#         for client in X:
#             for cut in client[item].drop_duplicates():  # 遍历客户端每个变量的每个切点，寻找分裂增益gain最大的切点并记录下来
#                 G_left = 0
#                 G_right = 0
#                 H_left = 0
#                 H_right = 0
#                 for client_j in X:  # 遍历每一个客户端
#                     if 0.001:  # 这里如果指定了min_child_sample则限制分裂后叶子节点的样本数都不能小于指定值
#                         if (client_j.loc[client_j[item] < cut].shape[0] < 10) \
#                                 | (client_j.loc[X[item] >= cut].shape[0] < 10):
#                             continue
#                     G_left += client_j.loc[client_j[item] < cut, 'g'].sum()
#                     G_right += client_j.loc[client_j[item] >= cut, 'g'].sum()
#                     H_left += client_j.loc[client_j[item] < cut, 'h'].sum()
#                     H_right += client_j.loc[client_j[item] >= cut, 'h'].sum()
#                 if 0.001:
#                     if (H_left < 0.001) | (H_right < 0.001):
#                         continue
#                 gain = G_left ** 2 / (H_left + 1) + G_right ** 2 / (H_right + 1)
#                 gain = gain - (G_left + G_right) ** 2 / (H_left + H_right + 1)
#                 gain = gain / 2 - 0
#                 if gain > max_gain:
#                     best_var, best_cut = item, cut
#                     max_gain = gain
#                     G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
#     if best_var is None or max_gain <= 0.00001:
#         return None
#     else:
#         all_client_left = []
#         for client in X:
#             id_left = client.loc[client[best_var] < best_cut].index.tolist()
#             all_client_left.append(client[id_left])
#         # w_left = - G_left_best / (H_left_best + 1)
#         all_client_right = []
#         for client in X:
#             id_right = X.loc[X[best_var] >= best_cut].index.tolist()
#             all_client_right.append(client[id_right])
#         # w_right = - G_right_best / (H_right_best + 1)
#         # w[id_left] = w_left
#         # w[id_right] = w_right
#         tree = {(best_var, best_cut): {}}
#         tree[(best_var, best_cut)]['left'] = xgb_cart_tree_server(all_client_left, depth + 1)  # 递归左子树
#         tree[(best_var, best_cut)]['right'] = xgb_cart_tree_server(all_client_right, depth + 1)  # 递归右子树
#     return tree


if __name__ == '__main__':
    '''
    csv_path_p = data_set.File_Upset
    csv_path = data_set.File_Train
    save_dir = 'Part_Data'
    # 执行打乱操作
    Data_Upset.get_label_data()
    # 对数据集进行训练集和测试集的分割
    T_C_Part.Tc_Part().split_csv(csv_path_p)
    # 将训练集平均分配给各用户机
    Data_Part.PyCSV().split_csv(csv_path, save_dir)
    '''

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
    all_client_X = []  # 所有客户端的列表X
    all_client_Y = []  # 所有客户端的列表Y

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
        client_file = 'Part_Data/Data_Train_' + str(i + 1) + '.csv'  # 不加cnn
        # client_file_cnn = 'myModels/data_transformed/tensor_' + str(i + 1) + '.csv'
        df = pd.read_csv(client_file)
        # df = pd.read_csv(client_file)
        X_train = df[df.columns[:-1].tolist()]
        Y_train = df[df.columns[-1]]
        # if X_train.shape[0] != Y_train.shape[0]:
        #     raise ValueError('X and Y must have the same length!')
        X = X_train.reset_index(drop=True)
        # Y = Y_train.values
        # y_hat = np.array([data_set.base_score] * Y.shape[0])
        # t0 = time.time()
        # gi = _grad('logistic', y_hat, Y)
        # hi = _hess('logistic', y_hat, Y)
        # assert type(gi) == np.ndarray
        # assert type(hi) == np.ndarray
        # X['g'] = gi
        # X['h'] = hi
        all_client_X.append(X)
        all_client_Y.append(Y_train)
        # gi_single_len = min(gi_single_len, gi.shape[0])
        # hi_single_len = min(hi_single_len, hi.shape[0])
        # client_hi.append(hi)
        # client_gi.append(gi)

    roc_auc = XGBoost_train.server_train(all_client_X, all_client_Y)
    print(roc_auc)
    # 计算平均的gi和hi
    # sum_gi = pd.Series(np.zeros(shape=(gi_single_len,)))
    # sum_hi = pd.Series(np.zeros(shape=(hi_single_len,)))
    # for index, value in enumerate(sum_gi):
    #     for client_gi_value in client_gi:
    #         sum_gi[index] += client_gi_value[index]
    # for index, value in enumerate(sum_hi):
    #     for client_hi_value in client_hi:
    #         sum_hi[index] += client_hi_value[index]
    #
    # sum_gi = np.array(sum_gi)
    # sum_hi = np.array(sum_hi)
    # FML = FED_XGB(learning_rate=0.1,
    #               n_estimators=20  # 总共迭代次数，每进行一轮进行一次全局更新
    #               , max_depth=4, min_child_weight=0.2, gamma=0.03,
    #               objective='logistic')
    # FML.xgb_cart_tree_server(all_client_X)
