# run_get_metrics_bacon.py

import csv
import numpy as np
import pandas as pd
from confidence_errors import get_confidence_errors, get_ace_confidence
import json


hyperparameter_file = './output/hyperparameters.json'
with open(hyperparameter_file) as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

class_wts_list = jsonObject['class_wts_list']
seed_holdout = jsonObject['seed_holdout']
class_count = jsonObject['class_count']
epsilon_threshold = jsonObject['epsilon_threshold']


print("class_wts_list:",class_wts_list)
print("seed_holdout:",seed_holdout)
print("class_count:",class_count)
print("epsilon_threshold:",epsilon_threshold)

# Check to see if test set is balanced_bayes_conf
test_set_balance = (class_wts_list.count(class_wts_list[0]) == len(class_wts_list))
print("initial check of test set balance = ", test_set_balance)

#Below code will read list of seeds from file
list_of_seeds = []
with open("./output/seeds.txt", "r") as f:
    for line in f:
        list_of_seeds.append(int(line.strip()))


opt_list = []
opt_list.append(list_of_seeds[seed_holdout])

#Open File to write (as csv)
metrics_out = open('./output/metrics.csv', 'w')

#Write header
if test_set_balance == True:
    metrics_header = 'Seed,Loss,Acc,ECE_Soft,MCE_Soft,MCE_Count_Soft,MCE_MinCount_Soft,MCE_MaxCountSoft,ACE_Soft,ECE_Scal_Soft,MCE_Scal_Soft,MCE_Count_Scal_Soft,MCE_MinCountScal_Soft,MCE_MaxCount_Scal_Soft,ACE_Scal_Soft,ECE_BAL_BACON,MCE_BAL_BACON,MCE_BAL_BACON_Count,MCE_BAL_BACON_MinCount,MCE_BAL_BACON_MaxCount,ACE_BAL_BACON\n'

else:
    metrics_header = 'Seed,Loss,Acc,ECE_Soft,MCE_Soft,MCE_Count_Soft,MCE_MinCount_Soft,MCE_MaxCount_Soft,ACE_Soft,ECE_Scal_Soft,MCE_Scal_Soft,MCE_Count_Scal_Soft,MCE_MinCount_Scal_Soft,MCE_MaxCount_Scal_Soft,ACE_Scal_Soft,ECE_BAL_BACON,MCE_BAL_BACON,MCE_BAL_BACON_Count,MCE_BAL_BACON_MinCount,MCE_BAL_BACON_MaxCount,ACE_BAL_BACON,ECE_IMB_BACON,MCE_IMB_BACON,MCE_IMB_BACON_COUNT,MCE_IMB_BACON_MinCount,MCE_IMB_BACON_Max_Count,ACE_IMB_BACON\n'

metrics_out.write(metrics_header)


#Loop through cases
for seed in list_of_seeds:

    if seed == list_of_seeds[seed_holdout]:
        continue
    
    print('\n')
    print('Working seed #', seed, '\n')
    
    # Retrieve Loss, Accuracy
    fname = './output/seed'+str(seed)+'/best_loss_acc.txt'
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        nnperf = rows[1]

    # Compute ECE for Softmax, CIPCE
    m = class_count-1 
    bins = np.linspace(0,1,m)
        
    # Softmax ECE
    if test_set_balance == True:
        SM_df = pd.read_csv('./output/seed'+str(seed)+'/softmax.csv')
    else:
        SM_df = pd.read_csv('./output/seed'+str(seed)+'/imbalanced_softmax.csv')
 
    SM_stem = "SM"
    
    SM_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
    
    for j,item in enumerate(SM_df.iloc()):
        if item['Label'] == item['Predicted']:
            accur = 1
        else:
            accur = 0
        conf = item[SM_stem + str(int(item['Predicted']))]
        SM_frame.loc[j] = [accur,conf]
    
    SM_ECE, SM_MCE, SM_MCE_count, SM_MCE_Max_count, SM_MCE_Min_count = get_confidence_errors(SM_frame,class_count)
    SM_ACE = get_ace_confidence(epsilon_threshold, class_count, m, SM_df, SM_stem)

    print('Softmax test_set balance = ', test_set_balance)
    
    # Scaled Softmax ECE/MCE
    if test_set_balance == True:
        Scal_SM_df = pd.read_csv('./output/seed'+str(seed)+'/balanced_scaled_softmax.csv')
    else:
        Scal_SM_df = pd.read_csv('./output/seed'+str(seed)+'/imbalanced_scaled_softmax.csv')
 
    Scaled_SM_stem = "SM"
    
    Scaled_SM_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
    
    for j,item in enumerate(Scal_SM_df.iloc()):
        if item['Label'] == item['Predicted']:
            accur = 1
        else:
            accur = 0
        conf = item[Scaled_SM_stem + str(int(item['Predicted']))]
        Scaled_SM_frame.loc[j] = [accur,conf]
    


    Scal_SM_ECE, Scal_SM_MCE, Scal_SM_MCE_count, Scal_SM_MCE_Max_count, Scal_SM_MCE_Min_count = get_confidence_errors(Scaled_SM_frame,class_count)
    Scal_SM_ACE = get_ace_confidence(epsilon_threshold, class_count, m, Scal_SM_df, Scaled_SM_stem)

    print('Scaled Softmax test_set balance = ', test_set_balance)
 
	# BACON ECE/MCE    
    Bac_stem = "P_b"

	
    if test_set_balance == True:
        Bal_Bac_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
        bal_bayes_prob_df = pd.read_csv('./output/seed'+str(seed)+'/balanced_bayes_conf.csv') 
        for j,item in enumerate(bal_bayes_prob_df.iloc()):
            if item['Label'] == item['Predicted']:
                accur = 1
            else:
                accur = 0
            conf = item[Bac_stem + str(int(item['Predicted']))]
            Bal_Bac_frame.loc[j] = [accur,conf]


        BAL_BAC_ECE, BAL_BAC_MCE, BAL_MCE_count, BAL_Max_count, BAL_Min_count = get_confidence_errors(Bal_Bac_frame,class_count)
        BAL_BAC_ACE = get_ace_confidence(epsilon_threshold, class_count, m, bal_bayes_prob_df, Bac_stem)

    else:
        Bal_Bac_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
        bal_bayes_prob_df = pd.read_csv('./output/seed'+str(seed)+'/balanced_bayes_conf.csv') 
        for j,item in enumerate(bal_bayes_prob_df.iloc()):
            if item['Label'] == item['Predicted']:
                accur = 1
            else:
                accur = 0
            conf = item[Bac_stem + str(int(item['Predicted']))]
            Bal_Bac_frame.loc[j] = [accur,conf]


        BAL_BAC_ECE, BAL_BAC_MCE, BAL_MCE_count, BAL_Max_count, BAL_Min_count = get_confidence_errors(Bal_Bac_frame,class_count)
        BAL_BAC_ACE = get_ace_confidence(epsilon_threshold, class_count, m, bal_bayes_prob_df, Bac_stem)	
	
        Imb_Bac_frame = pd.DataFrame(columns=["Accuracy","Confidence"])
        Imb_bayes_prob_df = pd.read_csv('./output/seed'+str(seed)+'/imbalanced_bayes_conf.csv') 
        for j,item in enumerate(Imb_bayes_prob_df.iloc()):
            if item['Label'] == item['Predicted']:
                accur = 1
            else:
                accur = 0
            conf = item[Bac_stem + str(int(item['Predicted']))]
            Imb_Bac_frame.loc[j] = [accur,conf] 
        		
        IMB_BAC_ECE, IMB_BAC_MCE, IMB_BAC_MCE_count, IMB_Max_count, IMB_Min_count = get_confidence_errors(Imb_Bac_frame,class_count)
        IMB_BAC_ACE = get_ace_confidence(epsilon_threshold, class_count, m, Imb_bayes_prob_df, Bac_stem)     


    print('Output record test set balance = ', test_set_balance)
    if test_set_balance == True:
        metric_record = str(seed) + ',' + str(nnperf[1]) + ',' + str(nnperf[2]) + ',' + str(SM_ECE) + ',' + str(SM_MCE) + ',' + str(SM_MCE_count) + ',' + str(SM_MCE_Min_count) + ',' + str(SM_MCE_Max_count) + ',' + str(SM_ACE) + ',' + str(Scal_SM_ECE) + ',' + str(Scal_SM_MCE) + ',' + str(Scal_SM_MCE_count) + ',' + str(Scal_SM_MCE_Min_count) + ',' + str(Scal_SM_MCE_Max_count) + ',' + str(Scal_SM_ACE) + ',' + str(BAL_BAC_ECE) + ',' + str(BAL_BAC_MCE) + ',' + str(BAL_MCE_count) + ',' + str(BAL_Min_count) + ',' + str(BAL_Max_count) + ',' + str(BAL_BAC_ACE) + '\n'

    else:
        metric_record = str(seed) + ',' + str(nnperf[1]) + ',' + str(nnperf[2]) + ',' + str(SM_ECE) + ',' + str(SM_MCE) + ',' + str(SM_MCE_count) + ',' + str(SM_MCE_Min_count) + ',' + str(SM_MCE_Max_count) + ',' + str(SM_ACE) + ',' + str(Scal_SM_ECE) + ',' + str(Scal_SM_MCE) + ',' + str(Scal_SM_MCE_count) + ',' + str(Scal_SM_MCE_Min_count) + ',' + str(Scal_SM_MCE_Max_count) + ',' + str(Scal_SM_ACE) + ',' + str(BAL_BAC_ECE) + ',' + str(BAL_BAC_MCE) + ',' + str(BAL_MCE_count) + ',' + str(BAL_Min_count) + ',' + str(BAL_Max_count) + ',' + str(BAL_BAC_ACE) + ',' + str(IMB_BAC_ECE) + ',' + str(IMB_BAC_MCE) + ',' + str(IMB_BAC_MCE_count) + ',' + str(IMB_Min_count) + ',' + str(IMB_Max_count) + ',' + str(IMB_BAC_ACE) + '\n'

    
    #print('metric_record', metric_record)
    metrics_out.write(metric_record)
    
# Close File
metrics_out.close()
