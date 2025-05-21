# run_script_compute_bacon.py

from get_angles import get_angles
from get_cauchy import get_cauchy
from get_params import get_params
from bayesian_confidence import bayesian_confidence

import json
import torch

import random
import numpy as np
import pandas as pd

import os


if __name__ == "__main__":

    hyperparameter_file = './output/hyperparameters.json'
    datasets_file = './datasets'
    save_folder = './output/'
    file_list = [datasets_file,save_folder]

    #Ensure files have a home
    for file in file_list:
        if os.path.exists(file) == False:
            os.mkdir(file)

    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    #Load hyperparameters from json file
    classes = jsonObject['classes']
    class_wts_list = jsonObject['class_wts_list']
    class_count = int(jsonObject['class_count'])
    seed_holdout = int(jsonObject['seed_holdout'])
    delta = float(jsonObject['delta'])

    balanced_wts_list = [1,1,1,1,1,1,1,1,1,1]


    #print("Parameters Successfully Loaded")
    print("classes:",classes)
    print("class_wts_list",class_wts_list)
    print("class_count:",class_count)
    print("seed_holdout:",seed_holdout)
    print("delta:",delta)

    if len(classes) != class_count:
        raise ValueError("Length of Classes at",len(classes),"does not equal the class_count variable at",class_count,"!!!")

    # Check to see if test set is balanced_bayes_conf
    test_set_balance = (class_wts_list.count(class_wts_list[0]) == len(class_wts_list))
    print("initial check of test set balance = ", test_set_balance)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('* * * * * * * * * * * * * * * *')
    print('Device Loaded')
    print('device = ', device)
    print('* * * * * * * * * * * * * * * *')



    #Below code will read list of seeds from file
    list_of_seeds = []
    with open("./output/seeds.txt", "r") as f:
        for line in f:
            list_of_seeds.append(int(line.strip()))

    # Loop through seeds
    for j in list_of_seeds:
        print("seed = ",j, '\n')
        print("seed holdout = ", list_of_seeds[seed_holdout],'\n')
        if j == list_of_seeds[seed_holdout]:
        
            continue
        
        random.seed(j)
        np.random.seed(j)
        torch.manual_seed(j)
        torch.cuda.manual_seed(j)
        torch.cuda.manual_seed_all(j)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        # This is needed for CUDA to run deterministically
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        
        print(j,'\n')

        print("class_wts_list:",class_wts_list)

        # Create Directory
        savedir = save_folder + 'seed' + str(j) +'/'
        if savedir is not None:
            try:
                os.mkdir(savedir)
            except FileExistsError: n=1

        dev_file_name = savedir+'dev_angles.csv'
        dev_angles = pd.read_csv(dev_file_name)
        cauchy_file = savedir+'cauchy_values.npy'
        lognorm_file = savedir+'lognorm_values.npy'
        cauchy_values,lognorm_values=get_params(dev_angles,hyperparameter_file,save_file_cauchy=cauchy_file,save_file_lognorm=lognorm_file)

        if test_set_balance == True:

            test_file_name = savedir+'test_angles.csv'
            test_angles = pd.read_csv(test_file_name)

            bayes_file = savedir + 'balanced_bayes_conf.csv'

            bayes_conf = bayesian_confidence(test_angles, cauchy_values, lognorm_values, delta, hyperparameter_file, balanced_wts_list, save_file=bayes_file, device=device)

        else:
            print('Saving result for imbalanced test set\n')
            test_file_name = savedir+'imbalanced_test_set.csv'
            test_angles = pd.read_csv(test_file_name)
            
            balanced_bayes_file = savedir+'balanced_bayes_conf.csv'

            bal_bayes_conf=bayesian_confidence(test_angles,cauchy_values,lognorm_values, delta, hyperparameter_file, balanced_wts_list,save_file=balanced_bayes_file, device=device)

            imbalanced_bayes_file = savedir+'imbalanced_bayes_conf.csv'

            imb_bayes_conf=bayesian_confidence(test_angles,cauchy_values,lognorm_values, delta, hyperparameter_file, class_wts_list,save_file=imbalanced_bayes_file, device=device)







