# 该文件负责储存大部分参数以备调用
# 数据参数维度数
num_item = 29
# 客户端数量（需要数据分割的份数）
num_user = 50
# 进行数据提取的文件路径
File_Name = 'Data/data_map10/creditcard1_train.csv'
# 少数类样本名称
File_Less = 'Data_Less.csv'
# 数据进行SMOTE过采样后的文件名称
File_Smote = 'Data_Smote/Data_Smote.csv'
# 打乱顺序后的文件路径
File_Upset = 'Part_Data/Data_Upset.csv'
# 分割后训练集路径
File_Train = 'Part_Data/Data_Trian.csv'
# 分割后测试集路径
File_Check = 'Data_Check/Data_Test.csv'
# 训练集比例
Train_Set = 0.8
User_w = ['0.01']
