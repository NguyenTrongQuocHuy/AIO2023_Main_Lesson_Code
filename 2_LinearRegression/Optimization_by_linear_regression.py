import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pandas as pd
import math
import random

#---------------------------------------------Data_Collecting-------------------------------------------
dataset_folder      =   r'C:\Github_NTQH\AIO2023_Main_Lesson_Code\2_LinearRegression\dataset' 
dataset_file_name   =   r'\advertising.csv'
dataset_file_path   =   dataset_folder  + dataset_file_name
lst_data            =   np.genfromtxt(dataset_file_path,delimiter=',',skip_header=1).tolist()

#lst_data_part_1     =   [lst_data[i][1] for i in range(len(lst_data))]   
#lst_data_part_2     =   [lst_data[i][len(lst_data[0])-1] for i in range(len(lst_data))]

#lst_data            =   [[lst_data_part_1[i]]+[lst_data_part_2[i]] for i in range(len(lst_data_part_1))]
#print(lst_data)

#-------------------------------------------------------------------------------------------------------

#----------------------------------------------Pseudo_code_area-----------------------------------------
def initialize_parameter_weight_bias(number_of_parameter_):
    lst_w_b_            =   []
    for _ in range(number_of_parameter_):
        lst_w_b_.append(random.gauss(mu=0, sigma=0.1))
    return lst_w_b_

def compute_predict_value(model_lst_w_b_, lst_data_feature_):
    lst_predict_value_      =   []
    for i_ in range(len(lst_data_feature_)):
        lst_predict_value_.append(np.sum(np.multiply(model_lst_w_b_,lst_data_feature_[i_])))
    return  lst_predict_value_

def loss_function(lst_w_b_, lst_feature_, label_):
    predict_value_      =   np.sum(np.multiply(lst_w_b_,lst_feature_))
    loss_func_          =   (predict_value_ - label_)**2
    return loss_func_

def compute_derivative(lst_w_b_, lst_feature_, label_, epsilon_=1e-5): #as derivative definition
    lst_derivative_     =   []
    for i_ in range(len(lst_w_b_)):
        lst_w_b_next_point_         =   lst_w_b_.copy()
        lst_w_b_next_point_[i_]     =   lst_w_b_next_point_[i_] + epsilon_
        derivative_as_definition_   =   (loss_function(lst_w_b_next_point_, lst_feature_,label_) - loss_function(lst_w_b_, lst_feature_,label_))/(epsilon_)
        lst_derivative_.append(derivative_as_definition_)
    return lst_derivative_                                                                      #lst_derivative = [dLoss/dw1, dLoss/dw2, dLoss/dw3...dLoss/dwi, dLoss/db]

def processing_all_database(epoch_, lr_, ini_lst_w_b_, lst_data_feature_, lst_data_label_): # Training by All batch data
    for _ in range(epoch_):
        for i_ in range(len(lst_data_feature_)):
            lst_derivative_i_       =   compute_derivative(ini_lst_w_b_, lst_data_feature_[i_], lst_data_label_[i_])
            lst_w_b_new_            =   np.subtract(ini_lst_w_b_, np.multiply(lr_,lst_derivative_i_))
            ini_lst_w_b_            =   lst_w_b_new_
            loss_val_               =   loss_function(lst_w_b_new_, lst_data_feature_[i_], lst_data_label_[i_])   
    return ini_lst_w_b_, loss_val_
#-------------------------------------------------------------------------------------------------------
#########################################################################################################################################################################   D
#-------------------------------------------Initialize_pre-train_parameters-----------------------------                                                                #   O
w_b_quantity                =   len(lst_data[0])                                                                                                                        #    
ini_lst_w_b                 =   initialize_parameter_weight_bias(w_b_quantity)                     # then we have initialize weight and bias                            #   N
#ini_lst_w_b                 =   [1,2]                                                                                                                                  #   O
print(f'List ini weights and bias ={ini_lst_w_b}')                                                                                                                      #   T
#=======================================================================================================                                                                #
print(f'pls input the number of epoch for trainning this linear regression model')                                                                                      #   M
epoch                   =   int(input())                                                            # then we have number of epoch                                      #   O    
print(f'Ok! This model will be trained with {epoch} epochs')                                                                                                            #   D
#=======================================================================================================                                                                #   I
print('pls input the learning rate')                                                                                                                                    #   F
lr                      =   float(input())                                                          # then we have learning rate                                        #   Y
#=======================================================================================================                                                                #   !
lst_data_feature        =   [lst_data[i][0:(len(ini_lst_w_b)-1)] for i in range(len(lst_data))]                                                                         #   !
[lst_data_feature[i].append(int(1)) for i in range(len(lst_data_feature))]                          # add 1 for bias. then we have feature of input dataset             #   !
lst_data_label          =   [lst_data[i][(len(ini_lst_w_b)-1)] for i in range(len(lst_data))]       # then we have label of input dataset                               #
#=======================================================================================================                                                                #
#########################################################################################################################################################################


#--------------------------------------------MODEL_TRAINING---------------------------------------------
model_lst_w_b,model_loss           =   processing_all_database(epoch, lr, ini_lst_w_b, lst_data_feature, lst_data_label)
print(f'after train, we get model parameter {model_lst_w_b}\nAnd the loss of model is {model_loss}')
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------VISUALIZATION-----------------------------------------------
lst_predict_value       =   compute_predict_value(model_lst_w_b,lst_data_feature)
plt.plot(lst_data_feature,lst_data_label,c='blue',marker='*')
plt.plot(lst_data_feature,lst_predict_value,c='red',marker='o')
plt.show()
