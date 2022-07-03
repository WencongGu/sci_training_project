# 目前负责调试各部分是否能正常运行
import Data
import Data_Part
import Data_Upset
import T_C_Part
import XGBoost_train

if __name__ == '__main__':
    csv_path_p = Data.File_Upset
    csv_path = Data.File_Train
    save_dir = 'Part_Data'
    # 执行打乱操作
    Data_Upset.get_label_data()
    # 对数据集进行训练集和测试集的分割
    T_C_Part.Tc_Part().split_csv(csv_path_p)
    # 将训练集平均分配给各用户机
    Data_Part.PyCSV().split_csv(csv_path, save_dir)

    # 客户端训练
    users = Data.num_user
    client_gi = []
    client_hi = []
    ave_gi = 0
    ave_hi = 0
    # 100轮联合更新
    for i in range(200):
        f1_score = []
        roc_auc = []
        ave_f1 = 0
        ave_roc_auc = 0
        for k in range(users):
            client_file = 'Part_Data/Data_Trian_' + str(k + 1) + '.csv'
            client_g_h = XGBoost_train.client_train(client_file, ave_gi, ave_hi, i)
            # 得到gi和hi以及其余评估值
            client_hi.append(client_g_h[1])
            client_gi.append(client_g_h[0])
            f1_score.append(client_g_h[2])
            roc_auc.append(client_g_h[3])
        # 计算平均的gi和hi
        ave_gi = 0
        ave_hi = 0
        for j in range(len(client_gi)):
            ave_gi += client_gi[j]
        for j in range(len(client_hi)):
            ave_hi += client_hi[j]
        for j in range(len(roc_auc)):
            # ave_f1 += f1_score[j]
            ave_roc_auc += roc_auc[j]
        # print("F1 score is: {}".format(ave_f1))
        print("AUC Score is: {}".format(ave_roc_auc))
        ave_hi = ave_hi / len(client_hi)
        ave_gi = ave_gi / len(client_gi)
