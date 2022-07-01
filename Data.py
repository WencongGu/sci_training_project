# 该文件负责储存大部分参数以备调用
# 数据参数维度数
num_item = 29
# 客户端数量（需要数据分割的份数）
num_user = 20
# 进行数据提取的文件路径
File_Name = 'Data/data_map10/creditcard1_train.csv'
# 打乱顺序后的文件路径
File_Upset = 'Part_Data/Data_Upset.csv'
# 分割后训练集路径
File_Train = 'Part_Data/Data_Trian.csv'
# 分割后检测集路径
File_Check = 'Data_Check/Data_Check.csv'
# 训练集比例
Train_Set = 0.8
User_w = ['0.01']
