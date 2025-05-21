# bacon.py

import sys
import numpy as np
from scipy.stats import cauchy, laplace_asymmetric, lognorm
import torch


def prob_vector(Phi, angle_node, class_count, delta, cauchy_values, lognorm_values,device='cpu'):

    """This function prepares a cauchy pdf for use by the bacon_prob function which calculates the bayesian estimate
    Arguments
        Phi:            list of floats, a list of the angles to calculate the cauchy vector 
        angle_node:     int, an integer indicating the node for which probabilities are being calculated
        class_count:    int, an integer indicating the amount of classes
        delta:          float, the path length for integrating the pdf
        cauchy_values:  np.array (class_count,class_count), a dictionary giving the parameter_values for cauchy pdf
        lognorm_values: np.array (class_count), a dictionary giving the scale value for cauchy pdf
    Returns
        prob_vector:     np.array (class_count,test_data_points), a matrix containing the requested pdf values from the 
                        Phi angles given above. Each individual value corresponds to the calculation
                            P(angle|class) which is required below in the bayes calculation.
    """

    
    prob_vector_list = []

    cauchy_loc = cauchy_values['loc']     
    cauchy_scale = cauchy_values['scale']

    lognorm_s = lognorm_values['s']
    lognorm_loc = lognorm_values['loc']
    lognorm_scale = lognorm_values['scale']

    phi_high = Phi + delta
    phi_low = Phi - delta
    
    for axis_node in range(class_count):
    
        if (axis_node == angle_node):


            cdf_lognorm_high = lognorm.cdf(phi_high,lognorm_s[axis_node],lognorm_loc[axis_node],lognorm_scale[axis_node])
            cdf_lognorm_low = lognorm.cdf(phi_low,lognorm_s[axis_node],lognorm_loc[axis_node],lognorm_scale[axis_node])
            lognorm_diff = cdf_lognorm_high - cdf_lognorm_low

            prob_vector_list.append(lognorm_diff)

        else:

            cdf_chy_high = cauchy.cdf(phi_high,cauchy_loc[axis_node,angle_node],cauchy_scale[axis_node,angle_node])
            cdf_chy_low = cauchy.cdf(phi_low,cauchy_loc[axis_node,angle_node],cauchy_scale[axis_node,angle_node])
            chy_diff = cdf_chy_high - cdf_chy_low

            prob_vector_list.append(chy_diff)

    prob_vector = np.array(prob_vector_list)


    return prob_vector


def bacon_prob(Phi, angle_node, class_count, class_wt,  delta, cauchy_values, lognorm_values,device='cpu'):
    """This function computes bayes rule for the given angles. Note, Bayes rule for an individual angle in our context is as follows.
    
        P(class|angle) =    P(angle|class) * P(class)
                            ------------------------ (Divide)
                            SUM{ P(angle|j) * P(j) }  < - - - - - Law of Total Probability
                        (Sum over all j classes)
        
    Arguments
        Phi:            list of floats, a list of the angles to calculate the cauchy vector 
        angle_node:     int, an integer indicating the output node for which probabilities are being predicted
        class_count:    int, an integer indicating the amount of classes
        class_wt:       list of floats, a list indicating the proportion each class is given
        delta:          float, the length in angles for integrating the pdf
        cauchy_values:  np.array (class_count,class_count), a dictionary giving the parameter values for cauchy pdf
        lognorm_values: np.array (class_count), a dictionary giving the parameter values for lognorm pdf
  
    Returns
        P_b:            list of floats, the length of the data points passed in. This function is used specifically to calculate all                                of the probabilities for a single class. Thus, each individual point is P(class|angle) as shown above, 
                            but function returns a list of probabilities.               
    """

    class_wt_arr = np.array(class_wt)
    prob_phi = prob_vector(Phi, angle_node, class_count, delta, cauchy_values, lognorm_values,device)

    a = class_wt[angle_node]*prob_phi[angle_node]
    b = np.dot(class_wt,prob_phi)
    P_b = np.divide(a,b)

    return P_b
