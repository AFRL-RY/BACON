# run_create_imbalanced_test_set.py

import numpy as np
import pandas as pd
import csv
import json

hyperparameter_file = 'hyperparameters.json'
with open(hyperparameter_file) as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()


class_count = jsonObject['class_count']
class_mask = jsonObject['class_wts_list']
seed_holdout = jsonObject['seed_holdout']

print("class_count:",class_count)
print("class_mask:",class_mask)
print("seed_holdout:",seed_holdout)

class_label = np.arange(0,class_count)

class_mask_array = np.array(class_mask)

seedcount = 30
input_dir_stem = './output/seed'
input_file_name = '/test_angles.csv'
SM_input_file_name = '/softmax.csv'
scaled_SM_input_file_name = '/balanced_scaled_softmax.csv'
test_output_file_name = '/imbalanced_test_set.csv'
SM_output_file_name = '/imbalanced_softmax.csv'
scaled_SM_output_file_name = '/imbalanced_scaled_softmax.csv'


#Below code will read list of seeds from file
list_of_seeds = []
with open("./output/seeds.txt", "r") as f:
    for line in f:
        list_of_seeds.append(int(line.strip()))


# Loop through each seed
for seed in list_of_seeds:
    print(list_of_seeds,'\n')
    print("seed = ", seed, '\n')
    if seed == list_of_seeds[seed_holdout]:
    #if i == seed_holdout:
        continue
    

    # Import softmax, scaled softmax, and test data file
    test_angles_df = pd.read_csv(input_dir_stem+str(seed)+input_file_name,index_col=[0])
    balanced_softmax_df = pd.read_csv(input_dir_stem+str(seed)+SM_input_file_name,index_col=[0])
    balanced_scaled_softmax_df = pd.read_csv(input_dir_stem+str(seed)+scaled_SM_input_file_name,index_col=[0])
    
    # Create index column in softmax,scaled softmax, and test data files
    test_angles_df = test_angles_df.index.to_frame(name='data_idx').join(test_angles_df)
    balanced_softmax_df = balanced_softmax_df.index.to_frame(name='data_idx').join(balanced_softmax_df)
    balanced_scaled_softmax_df = balanced_scaled_softmax_df.index.to_frame(name='data_idx').join(balanced_scaled_softmax_df)


    # Count frequency of data in file
    class_freq_list = []
    for count,class_idx in enumerate(class_label):
        class_idx = float(class_idx)
        class_freq = test_angles_df[test_angles_df.Label==class_idx].shape[0]
        class_freq_list.append(class_freq)

    # Get frequency of data for each class for balanced class distribution    
    balanced_class_dist = np.array(class_freq_list)
    print('balanced_class_dist = ',balanced_class_dist)

    # Create imbalanced class distribution
    imbalanced_class_dist = (class_mask_array*balanced_class_dist).astype(int)
    print('imbalanced_class_dist = ', imbalanced_class_dist)

    # Create blank data frame and add columns from input data
    column_names = test_angles_df.columns.values.tolist()
    imbalanced_test_df = pd.DataFrame(columns=column_names)
    
    # Loop through each class
    for count,class_idx in enumerate(class_label):

        class_idx_float = float(class_idx)
        # Sample from Balanced Data Set
        class_df = test_angles_df[test_angles_df['Label'] == class_idx_float].copy()
        class_df_sample = class_df.sample(n=imbalanced_class_dist[class_idx], axis='index')

        # Append to Blank Data Set
        imbalanced_test_df = pd.concat([imbalanced_test_df,class_df_sample],axis=0,ignore_index=True)

    # Save Test DataFrame to csv
    imbalanced_test_df.reset_index(drop=True, inplace=True)
    imbalanced_test_df.to_csv(input_dir_stem+str(seed)+test_output_file_name)
    
    # Create imbalanced Softmax DataFrame
    SM_column_names = balanced_softmax_df.columns.values.tolist()
    imbalanced_SM_df = pd.DataFrame(columns=SM_column_names)
    
    imbalanced_SM_df = balanced_softmax_df[balanced_softmax_df['data_idx'].isin(imbalanced_test_df['data_idx'])]
    
    #Save imbalanced Softmax DataFrame to csv
    imbalanced_SM_df.reset_index(drop=True, inplace=True)
    imbalanced_SM_df.to_csv(input_dir_stem+str(seed) + SM_output_file_name)    

    # Create imbalanced scaled Softmax DataFrame
    scaled_SM_column_names = balanced_scaled_softmax_df.columns.values.tolist()
    imbalanced_scaled_SM_df = pd.DataFrame(columns=scaled_SM_column_names)
    imbalanced_scaled_SM_df = balanced_scaled_softmax_df[balanced_scaled_softmax_df['data_idx'].isin(imbalanced_test_df['data_idx'])]
    
    #Save imbalanced scaled Softmax DataFrame to csv
    imbalanced_scaled_SM_df.reset_index(drop=True, inplace=True)
    imbalanced_scaled_SM_df.to_csv(input_dir_stem+str(seed) + scaled_SM_output_file_name)    
    

