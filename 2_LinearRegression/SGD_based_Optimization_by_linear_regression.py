#SGD Stochatis Gradient Descent
#Step1: Shuffle the data
#Step2: Select training sample and testing sample. Base on ratio: len(list_training_sample)) / len(list_data)

import numpy as np
import random
import Optimization_by_linear_regression_based_gradient_derivation
import matplotlib.pyplot as plt
dataset_folder      =   r'C:\Github_NTQH\AIO2023_Main_Lesson_Code\2_LinearRegression\dataset' 
dataset_file_name   =   r'\t32_1.csv'
dataset_file_path   =   dataset_folder  + dataset_file_name
lst_data            =   np.genfromtxt(dataset_file_path,delimiter=',',skip_header=1).tolist()


#################################################################################################################################################################################  ***   
#-------------------------------------------Initialize_pre-train_parameters-----------------------------                                                                        #  *** 
random.shuffle(lst_data)                                                                                            # shuffle input data                                        #  ***
#=======================================================================================================                                                                        #  ***
print('pls input ratio between training and testing')                                                                                                                           #  ***
ratio                   =   float(input())                                                                          # then we have ratio training/testing                       #  ***
training_data           =   lst_data[0:int(ratio*len(lst_data))]                                                                                                                #  ***
testing_data            =   lst_data[int(ratio*len(lst_data)):len(lst_data)]                                                                                                    #   III 
#=======================================================================================================                                                                        #   NNN   
w_b_quantity                =   len(training_data[0])                                                                                                                           #   III
ini_lst_w_b                 =   Optimization_by_linear_regression_based_gradient_derivation.initialize_parameter_weight_bias(w_b_quantity)    # ini weight and bias             #   TTT
print(f'List ini weights and bias ={ini_lst_w_b}')                                                                                                                              #   III
#=======================================================================================================                                                                        #   AAA
print(f'\nPls input the quantity of epoch for trainning this SGD model')                                                                                                        #   LLL
epoch                   =   int(input())                                                                            # then we have number of epoch                              #   III
print(f'\nWell! This SGD model will be trained with {epoch} epoch(s)')                                                                                                          #   ZZZ
#=======================================================================================================                                                                        #   EEE
print('\nPls input the learning rate')                                                                                                                                          #   ***  
lr                      =   float(input())                                                                          # then we have learning rate                                #   ***
#=======================================================================================================                                                                        #   *** 
lst_data_training_feature       =   [training_data[i][0:(len(ini_lst_w_b)-1)] for i in range(len(training_data))]                                                               #   ***
[lst_data_training_feature[i].append(int(1)) for i in range(len(training_data))]                                    # add 1 for bias. then we have feature of training dataset  #   ***  
lst_data_training_label         =   [training_data[i][(len(ini_lst_w_b)-1)] for i in range(len(training_data))]    # then we have label of training dataset                     #   ***
#=======================================================================================================                                                                        #   ***
lst_data_testing_feature        =   [testing_data[i][0:(len(ini_lst_w_b)-1)] for i in range(len(testing_data))]                                                                 #   ***
[lst_data_testing_feature[i].append(int(1)) for i in range(len(lst_data_testing_feature))]                          # add 1 for bias. then we have feature of testing dataset   #   *** 
lst_data_testing_label          =   [testing_data[i][(len(ini_lst_w_b)-1)] for i in range(len(testing_data))]       # then we have label of testing dataset                     #   ***
#=======================================================================================================                                                                        #   ***
# Now, we have data for training included as separated lists below:                                                                                                             #   ***
#       Training feature      =   lst_data_training_feature                                                                                                                     #   ***   
#       Training label        =   lst_data_training_label                                                                                                                       #   ***
#       Testing feature       =   lst_data_testing_feature                                                                                                                      #   ***      
#       Testing label         =   lst_data_testing_label                                                                                                                        #   ***
#################################################################################################################################################################################   ***

#-----------------------------------------------------------------------SGD_model_training_process-------------------------------------------------------------------------------
print('\nIf you want to train by the way to update one-by-one for every sample: type "obo"'
      '\nIf you want train by the way totalize then average for every epoch: type"avg"')
method_define           =   str(input())
if method_define        ==   "obo":
    print('Well! you chose "obo"')
    model_lst_w_b,model_loss,lst_loss_val           =   Optimization_by_linear_regression_based_gradient_derivation.training_all_database_obo(epoch, lr, ini_lst_w_b, lst_data_training_feature, lst_data_training_label)
    print(f'\nAfter train by method "obo",we get: \nmodel parameter {model_lst_w_b}\nAnd the loss of model is {model_loss}')
    
elif method_define      ==  "avg":
    print('Well! you chose "avg"')
    model_lst_w_b,model_loss,lst_loss_val           =   Optimization_by_linear_regression_based_gradient_derivation.training_all_database_average(epoch, lr, ini_lst_w_b, lst_data_training_feature, lst_data_training_label)
    print(f'\nAfter train by method "avg", we get: \nmodel parameter {model_lst_w_b}\nAnd the loss of model is {model_loss}')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------SGD_model_testing_process--------------------------------------------------------------------------------
#-------------------------------------------------------------------------------&_Visualization----------------------------------------------------------------------------------
lst_predict_value                   =   Optimization_by_linear_regression_based_gradient_derivation.compute_predict_value(model_lst_w_b, lst_data_testing_feature)


#https://matplotlib.org/stable/gallery/text_labels_and_annotations/align_ylabels.html#sphx-glr-gallery-text-labels-and-annotations-align-ylabels-py
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_demo2.html#sphx-glr-gallery-lines-bars-and-markers-scatter-demo2-py

fig, axs    =   plt.subplots(2, 2)
box = dict(facecolor='yellow', pad=5, alpha=0.2)

ax1         =   axs[0,0]
ax1.scatter([lst_data_testing_feature[_][0] for _ in range(len(lst_data_testing_feature))],lst_data_testing_label,c='blue',marker='*')
ax1.scatter([lst_data_testing_feature[_][0] for _ in range(len(lst_data_testing_feature))],lst_predict_value,c='red',marker='o')
ax1.set_xlabel('feature testing')
ax1.set_ylabel('label testing & predict')

ax2         =   axs[0,1]
ax2.plot([_ for _ in range(1,epoch+1,1)],lst_loss_val,  c='green') 
ax2.set_xlabel('Number of epoch')
ax2.set_ylabel('Loss value')

ax3         =   axs[1,0]
ax3.scatter([lst_data_training_feature[_][0] for _ in range(len(lst_data_training_feature))],lst_data_training_label,c='blue',marker='*')
ax3.set_xlabel('input training feature')
ax3.set_ylabel('input training label')

ax4         =   axs[1,1]
ax4.plot(lst_data_training_feature,lst_data_training_label,c='blue',marker='*')
ax4.set_xlabel('input training feature')
ax4.set_ylabel('input training label')

plt.show()