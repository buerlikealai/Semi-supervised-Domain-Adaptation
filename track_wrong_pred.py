import math
import numpy as np

def track_wrong_pred(prob,label_map,ground_truth,c):
    
  
    of_c_pred = label_map == c
    of_c_true = ground_truth == c # not all the pixels belong to class c will be predicted as c

    l = np.sum(of_c_pred)
    if l == 0:
        prob_of_c = 0
        true_pred_prob = 0

    else:
        prob_of_c = prob[of_c_pred]


        true_pred = np.logical_and(of_c_pred,of_c_true)
        true_pred_prob = prob[true_pred]
            

    return prob_of_c,true_pred_prob