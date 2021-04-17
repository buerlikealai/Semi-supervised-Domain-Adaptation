import math
import numpy as np

def sort_probs_clc(prob,label_map,least_confident_per):
    
    """
    prob is the output of softmax layer,
    
    label_map is the output of argmax function,
    
    least_confident_per is a pre-defined threshold which decides the reliability of predcited label, 
    
    all input are array, computation is carried out in CPU.
    
    Usage: by thresholding the predicted probability of each predicted label, only selecting top confident predictions for 
    retraining in Semi-supervised domain adaptation. 
   
    """
   
    prob_temp = prob

    for c in range(4):
        mask = np.zeros(prob.shape, dtype=np.bool)
        
        of_c = label_map == c # position of pixels belong to current class
        l = np.sum(of_c) # number of pixels belong to current class
        
        if l < 10: 
            replacement = -1
            label_map[of_c] = replacement
            
        else:
            replacement = -1
             
            prob_of_c = prob_temp[of_c] # position(probability) of pixels belong to current class

            sorted_probs = np.sort(prob_of_c) 
            
            # boundary between the least k% predictions and the rest
            threshold_temp = sorted_probs[np.int(np.ceil(least_confident_per*l))] 
            
            Cc = np.logical_and(prob_temp < threshold_temp, of_c)
          
            mask[Cc] = True  
            label_map[mask] = replacement 
    
    return label_map

def check_and_skip(prob_map,label_map,threshold):
    car_p = np.argwhere(label_map==4) # no cleansing of car pixels

    label_map_new = sort_probs_clc(prob_map,label_map,threshold)

    for i in car_p:
        label_map_new[i[0],i[1]]=4
       
    coverage = (256*256 - (np.sum(label_map_new==-1)))/(256*256)
      
    return np.copy(label_map_new),coverage
