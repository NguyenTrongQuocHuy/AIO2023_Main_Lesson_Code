import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_link =   r'C:\Github_NTQH\AIO2023_Main_Lesson_Code\1_EdgeDetection_by_gradient_derivation'
file_name   =   r'\forest&plane.jpg'
file_path   =   folder_link + file_name

lst_pic     =   cv2.imread(file_path,0).tolist()

height      =   len(lst_pic)
row         =   len(lst_pic[0])

def gradient_function(lst,x_,lr_):
    gradient_result_    =   (lst[x_+lr_] - lst[x_-lr_])/(lr_)
    return gradient_result_

lst_edge_result         =   [[0]*row for _ in range(height)]

for h in range(height):
    for r in range(1,row-1,1):
        #lst_edge_result[h][r]       =   ((gradient_function(lst_pic[h],r,1))+255)*(1/2)    #this is using scale method
        lst_edge_result[h][r]       =   np.abs(gradient_function(lst_pic[h],r,1))     #this is using absolute method
        
cv2.imwrite(folder_link+r'\forest&plane_edge.jpg',np.array(lst_edge_result))