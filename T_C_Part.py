# 该文件用于拆分训练集和测试集
import data_set
import csv
import os
import pandas as pd

class Tc_Part:
    # 开始进行拆分
    def __init__(self):
        self.total = None

    def split_csv(self, path):
        # 如果train.csv和check.csv存在就删除
        if os.path.exists(data_set.File_Train):
            os.remove(data_set.File_Train)
        if os.path.exists(data_set.File_Check):
            os.remove(data_set.File_Check)

        with open(path, 'r', newline='') as file:
            csvreader = csv.reader(file)
            i = 0
            # 获取原文件行数
            self.total = sum(1 for line in open(data_set.File_Name))
            # 获取训练集行数
            self.partline = (int)(self.total * data_set.Train_Set)
            self.encoding = 'utf-8'
            self.line_numbers = 0
            # print(self.total)
            df_iter = pd.read_csv(path,
                                  chunksize=self.partline,
                                  encoding=self.encoding)
            # 每次生成一个df，直到数据全部取完
            for df in df_iter:
                # 后缀从1开始
                i += 1
                # 统计数据总行数
                self.line_numbers += df.shape[0]
                # 设置切分后文件的保存路径
                if (i == 1):
                    save_filename = os.path.join(data_set.File_Train)
                else:
                    save_filename = os.path.join(data_set.File_Check)
                # 打印保存信息
                print(f"{save_filename} 已经生成！")
                # 保存切分后的数据
                df.to_csv(save_filename, index=False, encoding='utf-8', quoting=1)

            # 获取数据列名
            self.column_names = pd.read_csv(path, nrows=1).columns.tolist()
            print("切分完毕！")
        return None
