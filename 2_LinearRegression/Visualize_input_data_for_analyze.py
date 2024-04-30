import matplotlib.pyplot as plt
import numpy as np

dataset_folder      =   r'C:\Github_NTQH\AIO2023_Main_Lesson_Code\2_LinearRegression\dataset' 
dataset_file_name   =   r'\t06_1.csv'
dataset_file_path   =   dataset_folder  + dataset_file_name
lst_data            =   np.genfromtxt(dataset_file_path,delimiter=',',skip_header=1).tolist()

lst_feature         =   [lst_data[_][0] for _ in range(len(lst_data))]
lst_label           =   [lst_data[_][1] for _ in range(len(lst_data))]

plt.plot(lst_feature, lst_label, c='r')
plt.xlabel('feature')
plt.ylabel('label')
plt.show()