# bayesian_confidence.py

import json
import numpy as np
import pandas as pd
from bacon import bacon_prob
def bayesian_confidence(test_df, cauchy_values, lognorm_values, delta, hyperparameter_file, class_wts_list, save_file = None, device='cpu'):
    
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    class_count = int(jsonObject['class_count'])

    class_wts = np.array(class_wts_list)

    bays_prob = np.empty((class_count,len(test_df.index)))


    # Compute Bayesian Confidence
    angle_node_stem = 'Phi'
    for angle_node in range(class_count):

        angle_node_input = angle_node_stem + str(angle_node)



        bays_prob[angle_node] = bacon_prob(test_df[angle_node_input].to_numpy(),angle_node,class_count, class_wts, delta,
                                        cauchy_values,lognorm_values,device)

    bays_prob = bays_prob.swapaxes(0,1)

    bays_frame = pd.DataFrame(bays_prob,columns=['P_b'+str(i) for i in range(0,class_count)])
    frames = [test_df,bays_frame]
    test_Phi_SM_BS_frame = pd.concat(frames,axis=1)
    
    if save_file is not None:
        test_Phi_SM_BS_frame.to_csv(save_file,index=False)
    
    return test_Phi_SM_BS_frame
