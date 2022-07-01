# 目前负责调试各部分是否能正常运行
import Data_Part
import Data
import Data_Part
import Data_Upset
import T_C_Part

if __name__ == '__main__':
    csv_path_p = Data.File_Upset
    csv_path = Data.File_Train
    save_dir = 'Part_Data'
    # 执行打乱操作
    Data_Upset.get_label_data()
    T_C_Part.Tc_Part().split_csv(csv_path_p)
    Data_Part.PyCSV().split_csv(csv_path, save_dir)
