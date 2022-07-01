# 该文件执行数据读取工作
import Data
import numpy
import csv
import codecs


def load_data(File_Name):
    with open(File_Name, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            print(row)
