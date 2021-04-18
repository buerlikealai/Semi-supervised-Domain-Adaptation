import math
import numpy as np
from numba import jit
from numba import njit

@jit(nopython=True, cache=True)
def class_weights(label_map):
    for c in range(6):
        weight_c = np.empty((6,),dtype=np.float32)
        of_c = label_map == c
        of_c = np.ravel(of_c)
        temp = len(of_c)
        
        if temp == 0:
            pass
        else:
            #print('1/temp',1/temp)
            weight_c[c]= 1/temp
            
    #print('weight_c',weight_c)
    sum_weight_c = np.sum(weight_c) 
    #print('sum_weight_c',sum_weight_c)
    weight_matrix = np.ones((1,256,256),dtype=np.float32)   
    for i in range(256):
        for j in range(256):
            if label_map[i,j] == 0:
                weight_matrix[:,i,j] = weight_c[0]/sum_weight_c
                #print('weight_matrix[:,i,j]',weight_matrix[:,i,j])
            elif label_map[i,j] == 1:
                weight_matrix[:,i,j] = weight_c[1]/sum_weight_c
            elif label_map[i,j] == 2:
                weight_matrix[:,i,j] = weight_c[2]/sum_weight_c
            elif label_map[i,j] == 3:
                weight_matrix[:,i,j] = weight_c[3]/sum_weight_c  
            elif label_map[i,j] == 4:
                weight_matrix[:,i,j] = weight_c[4]/sum_weight_c
            elif label_map[i,j] == 5:
                weight_matrix[:,i,j] = weight_c[5]/sum_weight_c
            else:
                pass
    return weight_matrix