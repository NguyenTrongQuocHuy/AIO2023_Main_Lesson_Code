import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

def func_input(x_):
    y_       =  x_**2 + 6*x_ + 12 #x**4 + 10*x**3 + 10*x_**2 + 6*x_ + 12
    return y_
  
def gradient_func(x_, epsilon_):
    x_0_                =   x_ - epsilon_
    x_1_                =   x_ + epsilon_
    gradient_result_    =   (func_input(x_1_) - func_input(x_0_)) / (2*epsilon_) 
    return gradient_result_

x       =   np.random.randint(-20,0)
print('pls input the epsilon')
epsilon =   float(input())
print(x)
print('pls input an epoch')
epoch   =   int(input())
lst_y   =   [func_input(x)]
lst_x   =   [x]

for i in range(epoch):
    direction   =   np.sign(gradient_func(x,epsilon))
    x_new       =   x - (direction)*epsilon*np.abs(gradient_func(x,epsilon))
    x           =   x_new
    lst_y.append(func_input(x))
    lst_x.append(x)
    print(x)

print(f'y min = {lst_y[-1]}, at x = {lst_x[-1]}')
plt.plot([_ for _ in range(epoch+1)], lst_y, c='red', lw=2)
plt.xlabel('epoch')
plt.ylabel('y')
plt.show()