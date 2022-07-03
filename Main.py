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
    sum_gain = []
    # 100轮联合更新
    for i in range(100):
        for k in range(users):
            client_file = 'Part_Data/Data_Trian' + str(k + 1) + '.csv'
            sum_gain.append(XGBoost_train.client_train(client_file))
        # 计算最大信息增益
        max_gain = 0
        max_client = 0
        for j in len(sum_gain):
            if sum_gain[j] > max_gain:
                max_gain = sum_gain[j]
                max_client = j
