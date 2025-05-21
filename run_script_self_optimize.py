# run_script_self_optimize.py

from get_cauchy import get_cauchy
from get_params import get_params
from bayesian_confidence import bayesian_confidence
import json
import torch

import random
import numpy as np
import pandas as pd
import csv
from confidence_errors import get_confidence_errors, get_ace_confidence

import json
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
    print("Parameters Loading")
    classes = jsonObject['classes']
    class_count = int(jsonObject['class_count'])
    seed_holdout = jsonObject['seed_holdout']
    epsilon_threshold = jsonObject['epsilon_threshold']
    #delta_bacon = float(jsonObject['delta'])

    balanced_wts_list = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    m=class_count-1

    print("classes:",classes)
    print("class_count:",class_count)
    print("seed_holdout:",seed_holdout)

    #print("delta:",delta_bacon)

    if len(classes) != class_count:
        raise ValueError("Length of Classes at",len(classes),"does not equal the class_count variable at",class_count,"!!!")

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

    print("Seed = ", list_of_seeds[seed_holdout])



    j = list_of_seeds[seed_holdout]

        
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

    # Create Directory
    savedir = save_folder + 'seed' + str(j) +'/'
    if savedir is not None:
        try:
            os.mkdir(savedir)
        except FileExistsError: n=1




    dev_file_name = savedir+'dev_angles.csv'
    dev_angles = pd.read_csv(dev_file_name)



    test_file_name = savedir+'test_angles.csv'
    test_angles = pd.read_csv(test_file_name)   


    try_delta_bacon = np.arange(0.1,0.5,0.05)

    #Open File to write (as csv)
    metrics_out = open('./output/self_metrics.csv', 'w')

    metrics_header = 'delta_bacon,ECE_BACON,ACE_BACON,\n'

    metrics_out.write(metrics_header)

    cauchy_file = savedir + 'cauchy_values.npy'
    lognorm_file = savedir + 'lognorm_values.npy'
    cauchy_values, lognorm_values = get_params(dev_angles,hyperparameter_file, save_file_cauchy = cauchy_file, save_file_lognorm=lognorm_file)



    # Start Loop here 
    for delta in try_delta_bacon:

        bayes_file = savedir+'balanced_bayes_conf.csv'
        bayes_conf = bayesian_confidence(dev_angles, cauchy_values, lognorm_values, delta, hyperparameter_file, balanced_wts_list, save_file=bayes_file)


        # BACON ECE    
        bacon_prob_df = pd.read_csv(savedir + '/balanced_bayes_conf.csv')

        Bac_stem = "P_b"
    
        Bac_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
        for k,item in enumerate(bacon_prob_df.iloc()):
            if item['Label'] == item['Predicted']:
                accur = 1
            else:
                accur = 0

            conf = item[Bac_stem + str(int(item['Predicted']))]
            Bac_frame.loc[k] = [accur,conf]
    
             

        BACON_ECE = get_confidence_errors(Bac_frame,class_count)
        BACON_ACE = get_ace_confidence(epsilon_threshold, class_count, m, bacon_prob_df, Bac_stem)

        metric_record = str(delta) + ',' + str(BACON_ECE) + ',' + str(BACON_ACE) + '\n'


        metrics_out.write(metric_record)
    
    # Close File
    metrics_out.close()

    # Retrieve Metrics File
    CE_metrics = pd.read_csv('./output/self_metrics.csv')
    best_ECE = CE_metrics['ECE_BACON'].min()
    best_delta_ECE = CE_metrics.loc[CE_metrics['ECE_BACON'].idxmin()]['delta_bacon']
    best_ACE = CE_metrics['ACE_BACON'].min()
    best_delta_ACE = CE_metrics.loc[CE_metrics['ACE_BACON'].idxmin()]['delta_bacon']

    print("Best ECE = ", best_ECE, "\t", "delta = ", best_delta_ECE, "\n")
    print("Best ACE = ", best_ACE, "\t", "delta = ", best_delta_ACE, "\n")




