# get_params.py

import json
import numpy as np
import pandas as pd
from scipy.stats import cauchy, laplace_asymmetric, lognorm

def get_cauchy(angle_frame, hyperparameter_file, save_file=None):
    """
    WARNING ---> If the amount of items in a class are too small, not enough items will be in the training set, dev set, or test set
    in order for the code below to work, specifically when the angle_df is called in the nested four loops. The error that occurs is 
    that if one class has only 8 data items in the data set, the training set could be split up in such a way that the training set,
    dev set, or test set may be completely void of one of those examples. Thus, when angles are attempted to be calculated for the 
    certain set, it was never represented in the set and will throw an error. 
    """
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    class_count = int(jsonObject['class_count'])

    cauchy_loc = np.zeros((class_count,class_count))
    cauchy_scale = np.zeros((class_count, class_count))

    angle_df = angle_frame
    #Check for completeness of the data set
    labels = angle_df['Label']
    unique_set = set()
    for item in labels:
        unique_set.add(int(item))

    if len(unique_set) != class_count:
        raise ValueError("One of the classes is not represented in the training set. Check the unique set above this line to see which class is not represented. This usually occurs because of classes having very few examples")
    # i -> label, j -> predicted
    for i in range(class_count):
        angle_label = angle_df[angle_df['Label']==i]
        for j in range(class_count):
            bin_j = 'Phi'+str(j)
            cauchy_loc[i,j],cauchy_scale[i,j]=cauchy.fit(angle_label[bin_j])
    
    cauchy_values = {'loc': cauchy_loc,'scale': cauchy_scale}
    if save_file is not None:
        np.save(save_file,cauchy_values)
    return cauchy_values
    
def get_params(angle_frame, hyperparameter_file, save_file_cauchy=None, save_file_lognorm=None):
    """
    WARNING ---> If the amount of items in a class are too small, not enough items will be in the training set, dev set, or test set
    in order for the code below to work, specifically when the angle_df is called in the nested four loops. The error that occurs is 
    that if one class has only 8 data items in the data set, the training set could be split up in such a way that the training set,
    dev set, or test set may be completely void of one of those examples. Thus, when angles are attempted to be calculated for the 
    certain set, it was never represented in the set and will throw an error. 
    """
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    class_count = int(jsonObject['class_count'])

    cauchy_loc = np.zeros((class_count,class_count))
    cauchy_scale = np.zeros((class_count, class_count))

    lognorm_s = np.zeros(class_count)
    lognorm_loc = np.zeros(class_count)
    lognorm_scale = np.zeros(class_count)    
    
    angle_df = angle_frame

    #Check for completeness of the data set
    labels = angle_df['Label']
    unique_set = set()
    for item in labels:
        unique_set.add(int(item))

    if len(unique_set) != class_count:
        raise ValueError("One of the classes is not represented in the training set. Check the unique set above this line to see which class is not represented. This usually occurs because of classes having very few examples")
        
    # lbl -> label, out_node -> output_node
    for lbl in range(class_count):
        angle_label = angle_df[angle_df['Label']==lbl]
        for out_node in range(class_count):	
            bin_out_node = 'Phi'+str(out_node)
            if (lbl==out_node):
                lognorm_s[lbl], lognorm_loc[lbl], lognorm_scale[lbl] = lognorm.fit(angle_label[bin_out_node])
            else:
            	cauchy_loc[lbl,out_node],cauchy_scale[lbl,out_node]=cauchy.fit(angle_label[bin_out_node])
            	
    cauchy_values = {'loc': cauchy_loc, 'scale': cauchy_scale}
    lognorm_values = {'s': lognorm_s, 'loc': lognorm_loc, 'scale': lognorm_scale}
    
    if save_file_cauchy is not None:
    	np.save(save_file_cauchy,cauchy_values)
    if save_file_lognorm is not None:
        np.save(save_file_lognorm, lognorm_values)
    
    return cauchy_values, lognorm_values            
