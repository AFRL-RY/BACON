# get_cauchy.py

import json
import numpy as np
import pandas as pd
from scipy.stats import cauchy

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
