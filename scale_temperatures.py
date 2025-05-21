# scale_temperatures.py

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import os
import random
from get_scaled_softmax import get_Tscaled_softmax
from confidence_errors import get_confidence_errors
import matplotlib.pyplot as plt

from models import ResNet2
from models import Effnet

import get_softmax
from dataloaders import dataloaders

if __name__ == "__main__":
    
    hyperparameter_file = 'hyperparameters.json'
    datasets_file = './datasets'
    save_folder = './output/'
    file_list = [datasets_file,save_folder]
    
    #Ensure files have a home
    for file in file_list:
        if os.path.exists(file) == False:
            os.mkdir(file)

            
    # Import JSON parameters            
    with open(hyperparameter_file) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        
    seed_holdout = jsonObject['seed_holdout']
    model_type = jsonObject['model_type']
    class_count = int(jsonObject['class_count'])
    pretrained = bool(jsonObject['pretrained'])
    freeze_net = bool(jsonObject['freeze_net'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    # set seed to holdout set seed
    list_of_seeds = []
    with open("./output/seeds.txt","r") as f:
        for line in f:
            list_of_seeds.append(int(line.strip()))
            
    print("Seed = ", list_of_seeds[seed_holdout])
    
    opt_list = []
    opt_list.append(list_of_seeds[seed_holdout])
    
    j = opt_list[0]
    
    print('opt_list = ', opt_list, j)
    
    random.seed(j)
    np.random.seed(j)
    torch.manual_seed(j)
    torch.cuda.manual_seed(j)
    torch.cuda.manual_seed_all(j)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
    # Set save directory
    savedir = save_folder + 'seed' + str(j) +'/'
    if savedir is not None:
        try:
            os.mkdir(savedir)
        except FileExistsError: n=1

    # Import dataloaders
    train_loader,train_loader_2,dev_loader_train,dev_loader_post,test_loader = dataloaders(device)

    # import model
    model = ResNet2(model_type,class_count,pretrained,freeze_net).to(device)
    #model = Effnet(class_count).to(device)
    model_file = savedir + "best_model.pth"
    model.load_state_dict(torch.load(model_file))    
    
    # Initialize beta value
    beta = np.arange(0.4,0.8,0.01)
    print('beta = ', beta)
    
    NLL_list = []    
    
    scaled_softmax_output_file = savedir + 'Tscaled_softmax.csv'    
    
    for beta_inst in beta:
        T_softmax_vals = get_Tscaled_softmax(model,dev_loader_post,hyperparameter_file, beta_inst, device='cuda', file_name=None)
        
        SM_stem = "SM"
    
        T_SM_list = []
    
        for j,item in enumerate(T_softmax_vals.iloc()):
            if item['Label'] == item['Predicted']:
                accur = 1
            else:
                accur = 0
            conf = item[SM_stem + str(int(item['Label']))]
            T_SM_list.append(np.log(conf))
            
        T_SM_arr = np.array(T_SM_list)
        NLL_beta = - np.sum(T_SM_arr)
        NLL_list.append(NLL_beta)
        
        print('beta = ', beta_inst, '\n')
        print('NLL (beta) = ', NLL_beta, '\n')
    
    NLL_arr = np.array(NLL_list)

    p = np.polyfit(beta,NLL_arr,2)

    best_beta = -p[1]/(2*p[0])
    print('best beta = ',best_beta)
