# 该文件用于打乱初始数据集的行顺序
import pandas as pd
import os
import Data
from sklearn.utils import shuffle


def get_label_data():
    # 删除上次的结果
    if os.path.exists(Data.File_Upset):
        os.remove(Data.File_Upset)
    data = pd.read_csv(Data.File_Name, sep=',')
    data = shuffle(data)
    data.to_csv(Data.File_Upset, index=False, header=True)
