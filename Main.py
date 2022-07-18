# 目前负责调试各部分是否能正常运行
import numpy as np
import pandas

import data_set
import Data_Part
import Data_Upset
import T_C_Part
import XGBoost_train

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

    # 客户端训练
    users = data_set.num_user
    client_gi = []
    client_hi = []
    ave_gi = None
    ave_hi = None
    # 200轮联合更新
    for i in range(5):
        f1_score = []
        roc_auc = []
        ave_f1 = 0
        ave_roc_auc = 0
        hi_single_len = 0
        gi_single_len = 0
        for k in range(users):
            client_file = 'Part_Data/Data_Trian_' + str(k + 1) + '.csv'
            client_g_h = XGBoost_train.client_train(client_file, ave_gi, ave_hi, i)
            # 得到gi和hi以及其余评估值
            hi_single_len = client_g_h[1].shape[0]
            gi_single_len = client_g_h[0].shape[0]
            client_hi.append(client_g_h[1])
            client_gi.append(client_g_h[0])
            f1_score.append(client_g_h[2])
            roc_auc.append(client_g_h[3])
            assert type(client_g_h[0]) == pandas.Series
            assert type(client_g_h[1]) == pandas.Series
            assert type(client_g_h[2]) == list

        # 计算平均的gi和hi
        ave_gi = pandas.Series(np.zeros(shape=(gi_single_len,)))
        ave_hi = pandas.Series(np.zeros(shape=(hi_single_len,)))
        for index, value in enumerate(ave_gi):
            for client_gi_value in client_gi:
                ave_gi[index] += client_gi_value[index]
        for index, value in enumerate(ave_hi):
            for client_hi_value in client_hi:
                ave_hi[index] += client_hi_value[index]
        # for j in roc_auc:
        # ave_f1 += f1_score[j]
        # ave_roc_auc += j
        # print("F1 score is: {}".format(ave_f1))
        # print(f"AUC Score is: {ave_roc_auc}")
        ave_hi = ave_hi / len(client_hi)
        ave_gi = ave_gi / len(client_gi)
        ave_gi = np.array(ave_gi)
        ave_hi = np.array(ave_hi)
        print(ave_gi)
        print(ave_hi)
