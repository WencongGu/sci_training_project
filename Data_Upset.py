# 该文件用于打乱初始数据集的行顺序
import pandas as pd
import os
import data_set
from sklearn.utils import shuffle


def get_label_data():
    # 删除上次的结果
    if os.path.exists(data_set.File_Upset):
        os.remove(data_set.File_Upset)
    data = pd.read_csv(data_set.File_Name, sep=',')
    data = shuffle(data)
    data.to_csv(data_set.File_Upset, index=False, header=True)
