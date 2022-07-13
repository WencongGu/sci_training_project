# 该文件执行数据读取工作
import data_set
import numpy as np
import csv
import codecs


def load_data(File_Name):
    newindex = 0
    synthetic = np.zeros((sum(1 for line in open(data_set.File_Name)), 30))
    with open(File_Name, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            # print(row)
            synthetic[newindex] = row
            newindex += 1
    return synthetic
