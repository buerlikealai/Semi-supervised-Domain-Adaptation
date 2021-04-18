import math
import numpy as np

@jit(nopython=True)
def spatial_frequent(prob,images,least_prob):
    r,c = images.shape
    #print(images.shape)
    
    #initialize array
    spatial_frequent = np.empty((r,c),dtype=np.int64)
   
    """searching window """
    #define window dimension for estimation
       
    w_rw = 5
    w_cl = w_rw
       
    #define starting and end point in data for sliding window
    rw_stt = np.int( np.ceil(w_rw/2.) ) #ceil for defining largest integer
    #print('starting row indx',rw_stt)
   
    cl_stt = np.int( np.ceil(w_cl/2.) )
    #print('starting column indx',cl_stt)
   
    rw_stp = np.int( r - np.floor(w_rw/2.) ) # floor for defining smallest integer
    #print('ending row indx',rw_stp)
   
    cl_stp = np.int( c - np.floor(w_cl/2.) )
    #print('ending column indx',cl_stp)
       
       
    for rr in range(rw_stt,rw_stp+1): # serach window row wise
        #define extent of window
        #print('iteration in row',rr)
        from_rw = np.int( rr - np.ceil(w_rw/2.) ) # starts from indx=0
        #print('slide from the row indx of',from_rw)
       
        to_rw = np.int( rr + np.floor(w_rw/2.))
        #print('slide to the row indx of',to_rw)
           
        for cc in range(cl_stt, cl_stp+1):#search window colunm wise
            #define extent of window
            #print('iteration in column',cc)
           
            from_cl = np.int( cc - np.ceil(w_cl/2.) )
            #print('slide from the column indx of',from_cl)
           
            to_cl = np.int( cc + np.floor(w_cl/2.))
            #print('slide to the column indx of',to_cl)
               
            subview_label = (images[from_rw:to_rw,from_cl:to_cl])
            subview_prob = (prob[from_rw:to_rw,from_cl:to_cl])
            #print('selected windows',subview)
            #frequent_value = mode(subview.ravel)
            
            subview_label = subview_label.ravel()
            subview_prob = subview_prob.ravel()
            #print('subview',subview)
            
            """considering probability  """
            frequent_value = ((np.bincount(subview_label).argmax()))
            #print('frequent_value',frequent_value)
            of_c = subview_label == frequent_value
            of_c = np.ravel(of_c)
            prob_of_c = []
            for i in range(w_rw*w_rw):
                if of_c[i] == True:
                    temp = subview_prob[i]
                    prob_of_c.append(temp)
                else:
                    pass
            #print('prob_of_c',prob_of_c)
            if frequent_value == 10:
                frequent_value = frequent_value
            
            else: 
                total = 0
                for i in range(len(prob_of_c)):
                    total += prob_of_c[i]
                
                #print('total',total)
                #print('len(prob_of_c)',len(prob_of_c))
                #print('mean prob in sliding window:',total/len(prob_of_c))
                    
                if (total/len(prob_of_c))<least_prob:
                    frequent_value = 10
           
            """ convolve with structure window"""
            #print(frequent_value)
            spatial_frequent[rr,cc] = frequent_value
           
    spatial_frequent[0:rw_stt+1,:] = images[0:rw_stt+1,:]
    spatial_frequent[rw_stp-2:r+1,:] = images[rw_stp-2:r+1,:]
    spatial_frequent[:,0:cl_stt+1] = images[:,0:cl_stt+1]
    spatial_frequent[:,cl_stp:c+1] = images[:,cl_stp:c+1]
           
    return spatial_frequent

def check_and_skip(prob_map,label_map,threshold):
    car_p = np.argwhere(label_map==4) # no cleansing of car pixels
    
    label_map_new = spatial_frequent(prob_map,label_map,0.9)
    
    for i in car_p:
        label_map_new[i[0],i[1]]=4
            
    reject_p = np.argwhere(label_map_new==10)
    for i in reject_p:
        label_map_new[i[0],i[1]] = -1
       
    coverage = (256*256 - (np.sum(label_map_new==-1)))/(256*256)
      
    return np.copy(label_map_new),coverage