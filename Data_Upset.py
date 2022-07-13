# 该文件用于打乱初始数据集的行顺序
import pandas as pd
import os
import data_set
from sklearn.utils import shuffle


def get_label_data():
    # 删除上次的结果
    if os.path.exists(dataset.File_Upset):
        os.remove(dataset.File_Upset)
    data = pd.read_csv(dataset.File_Name, sep=',')
    data = shuffle(data)
    data.to_csv(dataset.File_Upset, index=False, header=True)
